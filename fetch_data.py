"""DefiLlama 链上数据爬取 CLI。

典型用法：

  # 列出 Ethereum 上 TVL 排名前 20 的池（不下载历史）
  python fetch_data.py --list --chain Ethereum --top 20

  # 用 pool_id 下载指定几个池的历史
  python fetch_data.py --pool-ids uuid1,uuid2,uuid3 --days 365 \
                       --pool-csv data/real_pools.csv \
                       --gas-csv  data/real_gas.csv

  # 自动选取按协议过滤的前 N 池历史
  python fetch_data.py --chain Ethereum --project aave-v3 --top 3 --days 365 \
                       --pool-csv data/real_pools.csv

  # 强制忽略缓存
  python fetch_data.py --pool-ids ... --no-cache

输出：
  - 项目格式的 pool CSV（含 timestamp / pool_id / apy / tvl / token_price / oracle_price）
  - 与 pool CSV 时间轴对齐的合成 gas CSV
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.onchain_fetcher import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    OnchainFetchError,
    build_pool_csv_from_defillama,
    list_top_pools,
    synthesize_gas_for_range,
)


# 演示用预设：3 池覆盖「稳定 / 中波动 / 高波动」三档
# 用 --demo 一键拉，能让回测看到真实的差异化策略表现
DEMO_POOL_SELECTIONS = [
    {
        "pool_id": "43641cf5-a92e-416b-bce9-27113d3c0db6",
        "display": "Maple_USDC",
        "note": "稳定币 + 5% APY",
    },
    {
        "pool_id": "359dd5cd-67a6-4f6a-83db-1edb301637e7",
        "display": "Vesper_ETH",
        "note": "ETH 高收益 ~10% APY，价格波动 ±50%",
    },
    {
        "pool_id": "747c1d2a-c668-4682-b9f9-296708a3dd90",
        "display": "Lido_stETH",
        "note": "stETH 低 APY 2.5%，价格跟随 ETH",
    },
]


# =================================================================
# 子命令实现
# =================================================================

def cmd_list(args: argparse.Namespace) -> None:
    pools = list_top_pools(
        chain=args.chain,
        project=args.project,
        min_tvl_usd=args.min_tvl,
        top_n=args.top,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
    )
    if not pools:
        print("[empty] 无符合条件的池")
        return
    print(f"{'pool_id':40s}  {'project':16s}  {'chain':12s}  {'symbol':24s}  {'tvl_usd':>14s}  apy(%)")
    for p in pools:
        apy_str = f"{p.apy:.2f}" if p.apy is not None else "n/a"
        print(f"{p.pool_id:40s}  {p.project:16s}  {p.chain:12s}  "
              f"{p.symbol:24s}  {p.tvl_usd:>14,.0f}  {apy_str}")


def cmd_download(args: argparse.Namespace) -> None:
    """根据 --demo / --pool-ids / 自动 top_n 选池，下载历史并写 CSV。"""
    # 1. 决定下载哪些池
    if args.demo:
        selections = [
            {"pool_id": p["pool_id"], "display": p["display"]}
            for p in DEMO_POOL_SELECTIONS
        ]
        print("[demo] 一键拉取演示组合：")
        for p in DEMO_POOL_SELECTIONS:
            print(f"  {p['display']:18s}  {p['note']}")
    elif args.pool_ids:
        ids = [s.strip() for s in args.pool_ids.split(",") if s.strip()]
        selections = []
        for pid in ids:
            # display 名：取末 8 位 + project 前缀（如果用户没给）
            selections.append({"pool_id": pid, "display": pid[:12]})
    else:
        pools = list_top_pools(
            chain=args.chain, project=args.project,
            min_tvl_usd=args.min_tvl, top_n=args.top,
            cache_dir=Path(args.cache_dir),
            use_cache=not args.no_cache,
        )
        if not pools:
            print("[abort] 自动选池为空，请放宽过滤或显式 --pool-ids")
            sys.exit(2)
        selections = [
            {"pool_id": p.pool_id, "display": f"{p.project}_{p.symbol[:8]}".replace("/", "-")}
            for p in pools
        ]
        print("[auto-select]")
        for s in selections:
            print(f"  {s['display']:30s}  pool_id={s['pool_id']}")

    # 2. 下载并写 pool CSV
    out_pool = Path(args.pool_csv)
    try:
        pool_df = build_pool_csv_from_defillama(
            selections=selections,
            output_path=out_pool,
            n_days=args.days,
            cache_dir=Path(args.cache_dir),
            use_cache=not args.no_cache,
        )
    except OnchainFetchError as e:
        print(f"[error] {e}")
        sys.exit(3)
    print(f"[save] pool CSV → {out_pool} ({len(pool_df)} rows, "
          f"{pool_df['pool_id'].nunique()} pools)")

    # 3. 合成对齐的 Gas CSV
    out_gas = Path(args.gas_csv) if args.gas_csv else None
    if out_gas is not None:
        ts_grid = pd.DatetimeIndex(sorted(pool_df["timestamp"].unique()))
        spike = []
        if args.gas_spike_factor and args.gas_spike_start is not None:
            spike.append((args.gas_spike_start, args.gas_spike_duration, args.gas_spike_factor))
        gas_df = synthesize_gas_for_range(
            ts_grid, spike_windows=spike, output_path=out_gas,
        )
        print(f"[save] gas CSV → {out_gas} ({len(gas_df)} rows; "
              f"spike={spike if spike else 'none'})")


# =================================================================
# 参数解析
# =================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DefiLlama 数据爬取")
    p.add_argument("--list", action="store_true", help="只列池（不下载历史）")
    p.add_argument("--chain", default="Ethereum")
    p.add_argument("--project", default=None, help="可选协议过滤，如 aave-v3 / curve-dex")
    p.add_argument("--min-tvl", type=float, default=1_000_000)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--demo", action="store_true",
                   help="一键拉取演示组合（稳定 USDC + 波动 ETH 池）；优先级最高")
    p.add_argument("--pool-ids", default=None,
                   help="逗号分隔的 pool_id 列表；提供时跳过自动选池")
    p.add_argument("--days", type=int, default=365, help="取末 N 天")
    p.add_argument("--pool-csv", default="data/real_pools.csv")
    p.add_argument("--gas-csv", default="data/real_gas.csv")
    p.add_argument("--gas-spike-start", type=int, default=None,
                   help="gas spike 起始 tick（基于 pool_df 时间网格）")
    p.add_argument("--gas-spike-duration", type=int, default=5)
    p.add_argument("--gas-spike-factor", type=float, default=None,
                   help="gas spike 倍数；不给则不注入 spike")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    if args.list:
        cmd_list(args)
    else:
        cmd_download(args)


if __name__ == "__main__":
    main()
