"""DefiLlama 链上数据爬取与缓存。

为什么选 DefiLlama：
  - 免费、无 auth；覆盖几千个池
  - /pools 列出全部池快照（含 pool_id / symbol / chain / project / tvlUsd / apy）
  - /chart/{pool_id} 返回单池历史时序（每日的 apy / tvlUsd）
  - APY 字段单位为 percent（例如 5.2 表示 5.2%），需要除以 100

为什么不引入 requests：
  - 只做简单 GET，stdlib `urllib` 够用
  - 减少依赖；离线环境只需把 cache 拷过去即可

设计：
  - 内存 + 磁盘双层缓存（cache_dir 默认 data/.cache/）
  - 网络错误 → 抛 OnchainFetchError，调用方自行决定是否回退合成数据
  - 所有时间戳统一为 UTC tz-naive，避免 pandas 时区运算坑
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# =================================================================
# 端点 / 缓存目录
# =================================================================

DEFILLAMA_POOLS_URL = "https://yields.llama.fi/pools"
DEFILLAMA_CHART_URL = "https://yields.llama.fi/chart/{pool_id}"
COINS_LLAMA_CHART_URL = "https://coins.llama.fi/chart/{chain}:{address}"

DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"

# 部分代币的 underlying = 0x0 占位符（如 Lido stETH 的 underlying 写成 ETH）；
# 在这种情况下回退到 symbol → 真实合约地址的硬编码映射。覆盖范围有限，
# 但已经够 demo 用；扩展时直接往 SYMBOL_TOKEN_ADDRESS 里加即可。
_ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
SYMBOL_TOKEN_ADDRESS: dict[str, tuple[str, str]] = {
    # symbol(upper) → (chain_lower, address)
    "STETH": ("ethereum", "0xae7ab96520de3a18e5e111b5eaab095312d7fe84"),
    "WSTETH": ("ethereum", "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0"),
    "WBETH": ("ethereum", "0xa2e3356610840701bdf5611a53974510ae27e2e1"),
    "WEETH": ("ethereum", "0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee"),
    "EZETH": ("ethereum", "0xbf5495efe5db9ce00f80364c8b423567e58d2110"),
    "RETH":  ("ethereum", "0xae78736cd615f374d3085123a210448e74fc6393"),
    # 稳定币显式标 None：build_pool_csv 看到 None 就直接 1.0，不发 HTTP
    "USDC": None, "USDT": None, "DAI": None, "USDS": None, "SUSDS": None,
    "FRAX": None, "LUSD": None, "GHO": None, "PYUSD": None, "USDE": None,
}


# =================================================================
# 异常
# =================================================================

class OnchainFetchError(Exception):
    """fetcher 层不可恢复错误（网络、解析、空数据等）。"""


# =================================================================
# 元数据值对象
# =================================================================

@dataclass(frozen=True, slots=True)
class PoolMeta:
    """DefiLlama /pools 响应中的单条池元数据。"""

    pool_id: str        # 唯一 uuid（用于查 /chart 历史）
    symbol: str         # 友好显示名，例如 'USDC-DAI'
    project: str        # 协议名，例如 'aave-v3'
    chain: str          # 'Ethereum' / 'Arbitrum' / ...
    tvl_usd: float
    apy: Optional[float]  # 当前 APY (percent)
    underlying_tokens: tuple = ()   # 基础代币地址列表


# =================================================================
# 工具：HTTP + 缓存
# =================================================================

def _http_get_json(url: str, timeout: float = 15.0) -> Any:
    """stdlib GET JSON。抛 OnchainFetchError 把网络/解析错误统一。"""
    req = urllib.request.Request(
        url, headers={"User-Agent": "defi-backtest/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise OnchainFetchError(f"HTTP 失败 {url}: {e}") from e
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise OnchainFetchError(f"JSON 解析失败 {url}: {e}") from e


def _cache_path(cache_dir: Path, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 仅保留字母数字下划线 + 连字符，防止文件名非法
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return cache_dir / f"{safe}.json"


def _load_cached(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("cache 读取失败 %s: %s", path, e)
        return None


def _save_cached(path: Path, data: Any) -> None:
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError as e:
        logger.warning("cache 写入失败 %s: %s", path, e)


# =================================================================
# /pools：列出顶池
# =================================================================

def list_top_pools(
    chain: Optional[str] = "Ethereum",
    project: Optional[str] = None,
    min_tvl_usd: float = 1_000_000,
    top_n: int = 20,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> List[PoolMeta]:
    """列出按 TVL 排序的顶池（可按链 / 协议过滤）。

    DefiLlama /pools 端点会被缓存到 cache_dir/pools_list.json；
    第二次调用同 chain/project 过滤将直接复用。
    """
    cache_file = _cache_path(cache_dir, "pools_list")
    raw = _load_cached(cache_file) if use_cache else None
    if raw is None:
        logger.info("[fetch] %s", DEFILLAMA_POOLS_URL)
        raw = _http_get_json(DEFILLAMA_POOLS_URL)
        _save_cached(cache_file, raw)

    pools = raw.get("data") if isinstance(raw, dict) else None
    if not pools:
        raise OnchainFetchError("DefiLlama /pools 返回空数据")

    filtered: List[PoolMeta] = []
    for p in pools:
        if chain and p.get("chain") != chain:
            continue
        if project and p.get("project") != project:
            continue
        tvl = float(p.get("tvlUsd") or 0)
        if tvl < min_tvl_usd:
            continue
        underlying = tuple(
            str(t).lower() for t in (p.get("underlyingTokens") or [])
        )
        filtered.append(PoolMeta(
            pool_id=str(p.get("pool")),
            symbol=str(p.get("symbol") or ""),
            project=str(p.get("project") or ""),
            chain=str(p.get("chain") or ""),
            tvl_usd=tvl,
            apy=float(p["apy"]) if p.get("apy") is not None else None,
            underlying_tokens=underlying,
        ))

    filtered.sort(key=lambda m: m.tvl_usd, reverse=True)
    return filtered[:top_n]


# =================================================================
# /chart/{pool_id}：单池历史
# =================================================================

def fetch_pool_history(
    pool_id: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """拉取单池的历史 APY / TVL 时序。

    返回 DataFrame：timestamp(datetime, tz-naive) / apy(decimal) / tvl(float)
    APY 字段从 DefiLlama 的 percent 单位转为 decimal（5.2 → 0.052）。
    """
    cache_file = _cache_path(cache_dir, f"chart_{pool_id}")
    raw = _load_cached(cache_file) if use_cache else None
    if raw is None:
        url = DEFILLAMA_CHART_URL.format(pool_id=pool_id)
        logger.info("[fetch] %s", url)
        raw = _http_get_json(url)
        _save_cached(cache_file, raw)

    rows = raw.get("data") if isinstance(raw, dict) else None
    if not rows:
        raise OnchainFetchError(f"DefiLlama /chart/{pool_id} 返回空历史")

    df = pd.DataFrame(rows)
    if "timestamp" not in df.columns or "apy" not in df.columns:
        raise OnchainFetchError(
            f"chart 响应缺少 timestamp/apy 列：{df.columns.tolist()}"
        )

    # DefiLlama 的快照时间戳在每池有秒级偏移；归一化到日级（floor 到 00:00），
    # 否则跨池 timestamp 永远不相等，build_asset_snapshots 会取不到交集。
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
          .dt.tz_convert(None)
          .dt.floor("D")
    )
    df["apy"] = df["apy"].astype(float) / 100.0          # percent → decimal
    if "tvlUsd" in df.columns:
        df["tvl"] = df["tvlUsd"].astype(float)
    else:
        df["tvl"] = float("nan")
    # 同一池同一天可能有多条（DefiLlama 偶发重复）→ 按 timestamp 去重保留末条
    df = (df[["timestamp", "apy", "tvl"]]
          .drop_duplicates(subset=["timestamp"], keep="last")
          .sort_values("timestamp")
          .reset_index(drop=True))
    return df


# =================================================================
# 批量构建项目 CSV 格式
# =================================================================

def find_pool_meta_by_id(
    pool_id: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> Optional[PoolMeta]:
    """从 /pools 缓存里找单个 pool 的元数据；缓存缺失时主动拉一次。"""
    cache_file = _cache_path(cache_dir, "pools_list")
    raw = _load_cached(cache_file) if use_cache else None
    if raw is None:
        raw = _http_get_json(DEFILLAMA_POOLS_URL)
        _save_cached(cache_file, raw)
    pools = raw.get("data") if isinstance(raw, dict) else None
    if not pools:
        return None
    for p in pools:
        if p.get("pool") == pool_id:
            underlying = tuple(
                str(t).lower() for t in (p.get("underlyingTokens") or [])
            )
            return PoolMeta(
                pool_id=pool_id,
                symbol=str(p.get("symbol") or ""),
                project=str(p.get("project") or ""),
                chain=str(p.get("chain") or ""),
                tvl_usd=float(p.get("tvlUsd") or 0),
                apy=float(p["apy"]) if p.get("apy") is not None else None,
                underlying_tokens=underlying,
            )
    return None


def _resolve_token_address(meta: PoolMeta) -> Optional[tuple[str, str]]:
    """决定该 pool 用哪个 token 的价格作为代表。

    返回 (chain_lower, address) 或 None（视为 1.0 稳定币）。

    决策顺序：
      1. 若 symbol 在 SYMBOL_TOKEN_ADDRESS 中显式映射为 None → 视为稳定 (1.0)
      2. 若 symbol 显式映射为有效 (chain, address) → 用它
      3. 若 underlying_tokens 非空且首项不是 0x0 → 用 (chain, address)
      4. 否则返回 None（fallback 到 1.0）
    """
    sym = meta.symbol.upper()
    if sym in SYMBOL_TOKEN_ADDRESS:
        return SYMBOL_TOKEN_ADDRESS[sym]
    # 也试一下 symbol 的子串（例如 'STETH-WETH' 命中 STETH）
    for known_sym, mapped in SYMBOL_TOKEN_ADDRESS.items():
        if known_sym in sym:
            return mapped
    # 用 underlying_tokens[0] 兜底
    if meta.underlying_tokens:
        first = meta.underlying_tokens[0]
        if first and first != _ZERO_ADDRESS:
            return (meta.chain.lower(), first)
    return None


def _resolve_pool_token_price(
    pool_id: str,
    pool_history_df: pd.DataFrame,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> pd.Series:
    """对单个 pool 解析 token 地址并拉历史价格，对齐到 pool_history_df.timestamp。

    失败/稳定币 → 返回长度等于 df 的常数 1.0 Series。
    成功 → 返回对齐后的价格 Series。
    """
    n = len(pool_history_df)
    fallback = pd.Series([1.0] * n, dtype=float)

    meta = find_pool_meta_by_id(pool_id, cache_dir=cache_dir, use_cache=use_cache)
    if meta is None:
        logger.warning("[price] pool_id=%s 元数据查不到，token_price 回退 1.0", pool_id)
        return fallback

    resolved = _resolve_token_address(meta)
    if resolved is None:
        # 已知稳定币或无法解析 → 直接 1.0，不浪费 HTTP
        logger.info(
            "[price] pool=%s (symbol=%s) 视为稳定 token，price=1.0",
            pool_id, meta.symbol,
        )
        return fallback

    chain, address = resolved
    try:
        # span 取 pool 历史天数 + buffer，确保覆盖
        span = max(int(len(pool_history_df) * 1.5), 60)
        price_df = fetch_token_price_history(
            chain=chain, address=address, span=span, period="1d",
            cache_dir=cache_dir, use_cache=use_cache,
        )
    except OnchainFetchError as e:
        logger.warning(
            "[price] pool=%s symbol=%s 价格拉取失败: %s; 回退 1.0",
            pool_id, meta.symbol, e,
        )
        return fallback

    # 用 timestamp 做 left-join，缺失填前值
    merged = pool_history_df[["timestamp"]].merge(
        price_df, on="timestamp", how="left",
    )
    merged["price"] = merged["price"].ffill().bfill()
    if merged["price"].isna().any():
        logger.warning(
            "[price] pool=%s symbol=%s 价格序列仍有 NaN，回退 1.0", pool_id, meta.symbol,
        )
        return fallback

    # 归一化到「期初 = 1.0」便于 NAV 比较（保留相对变化幅度）
    p0 = float(merged["price"].iloc[0])
    if p0 <= 0:
        return fallback
    normalized = merged["price"].astype(float) / p0
    logger.info(
        "[price] pool=%s symbol=%s 真实价格 OK；范围 %.4f ~ %.4f（归一化）",
        pool_id, meta.symbol,
        float(normalized.min()), float(normalized.max()),
    )
    return normalized.reset_index(drop=True)


def fetch_token_price_history(
    chain: str,
    address: str,
    span: int = 365,
    period: str = "1d",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """从 coins.llama.fi 拉单 token 的历史价格。

    返回 DataFrame：timestamp (datetime, tz-naive, floored to day) / price (float)
    """
    cache_file = _cache_path(cache_dir, f"price_{chain}_{address}_{span}_{period}")
    raw = _load_cached(cache_file) if use_cache else None
    if raw is None:
        base_url = COINS_LLAMA_CHART_URL.format(chain=chain, address=address)
        url = f"{base_url}?span={span}&period={period}"
        logger.info("[fetch] %s", url)
        raw = _http_get_json(url)
        _save_cached(cache_file, raw)

    key = f"{chain}:{address}"
    coin = (raw.get("coins") or {}).get(key)
    if not coin:
        raise OnchainFetchError(f"coins.llama.fi 无 {key} 数据")
    prices = coin.get("prices") or []
    if not prices:
        raise OnchainFetchError(f"coins.llama.fi {key} 价格历史为空")

    df = pd.DataFrame(prices)
    # timestamps 是 unix 秒，需要先转 datetime 再 floor 到 day
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert(None).dt.floor("D")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "price"]]


def build_pool_csv_from_defillama(
    selections: List[Dict[str, str]],
    output_path: Optional[Path] = None,
    n_days: Optional[int] = 365,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    """从 DefiLlama 多池历史合成项目格式的 pool CSV。

    selections: [{"pool_id": "<uuid>", "display": "Aave-USDC"}, ...]
    output_path: 若提供，结果会落盘
    n_days: 仅取最近 n_days 个点；None 表示全量

    返回 DataFrame，列顺序：timestamp / pool_id / apy / tvl / token_price / oracle_price
    """
    if not selections:
        raise ValueError("selections 不能为空")

    all_dfs: List[pd.DataFrame] = []
    for sel in selections:
        pid = sel["pool_id"]
        display = sel.get("display") or pid[:8]
        df = fetch_pool_history(pid, cache_dir=cache_dir, use_cache=use_cache)
        if n_days is not None and n_days > 0:
            df = df.tail(n_days)
        df = df.copy().reset_index(drop=True)   # 关键：重置索引避免后续 Series 对齐失败
        df["pool_id"] = display

        # 尝试拉真实 token 价格；失败/无映射时降级到 1.0
        token_price = _resolve_pool_token_price(
            pid, df, cache_dir=cache_dir, use_cache=use_cache,
        )
        # 用 .values 而非 Series 避免索引对齐踩坑
        df["token_price"] = token_price.values if hasattr(token_price, "values") else token_price
        df["oracle_price"] = df["token_price"]

        all_dfs.append(df)

    pool_df = pd.concat(all_dfs, ignore_index=True)[
        ["timestamp", "pool_id", "apy", "tvl", "token_price", "oracle_price"]
    ]
    pool_df = pool_df.sort_values(["pool_id", "timestamp"]).reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pool_df.to_csv(output_path, index=False)
        logger.info("[save] pool csv → %s (%d rows)", output_path, len(pool_df))

    return pool_df


# =================================================================
# 合成 Gas（DefiLlama 不提供）
# =================================================================

def synthesize_gas_for_range(
    timestamps: pd.DatetimeIndex,
    base_fee: float = 1e-7,
    priority_fee: float = 5e-8,
    spike_windows: Optional[List[tuple[int, int, float]]] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """根据给定时间网格合成 Gas DataFrame。

    spike_windows: [(start_tick, duration, factor), ...]
    output_path:   若提供则落盘 CSV
    """
    n = len(timestamps)
    bf = [base_fee] * n
    pf = [priority_fee] * n
    if spike_windows:
        for start, duration, factor in spike_windows:
            for i in range(start, min(start + duration, n)):
                bf[i] *= factor
                pf[i] *= factor

    df = pd.DataFrame({
        "timestamp": timestamps,
        "base_fee": bf,
        "priority_fee": pf,
    })
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df
