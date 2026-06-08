"""示例数据生成器：合成 pool / gas DataFrame，供 UI 与 run_example 使用。

V2 改进（解决 MDD=0 / Sharpe 爆掉问题）：
  - token_price 不再恒定为 1.0；按每池配置的 token_vol 做几何 Brownian 走（GBM）
  - oracle_price 独立游走，仍提供 LVR 来源
  - APY 在 crash 窗口内可短暂为负（更激烈的应激）
  - pool_a / pool_b / pool_c 三池被赋予不同的 token 波动率，分别代表
    「稳定币型 / 半稳定 / 高波动」三类资产

数据特征（默认参数）：
  - 3 池：pool_A / pool_B / pool_C，APY 基线分别 5% / 7% / 9%
  - 365 天日级时序
  - 闪崩注入到 pool_B 末段（200~204 天）
  - Gas 在 150~154 天注入 5× 暴涨
  - 每池独立 token_price 漂移，让 NAV 反映价格变化 → 真实回撤

字段约定：
  - 所有 price/fee/apy 字段为 float；落到 build_asset_snapshots 时通过
    Decimal(str(x)) 安全转换
  - oracle_price 是「链上预言机价格」，token_price 是池内成交价
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


_DEFAULT_POOL_IDS = ["pool_A", "pool_B", "pool_C"]

# 每池的 token 价格年化波动率（默认）
# pool_A 稳定币型 (~2%) / pool_B 半稳定 (~8%) / pool_C 适度波动 (~18%)
# 在 365 天上对应的预期最大回撤大致为 1% / 8% / 22%，体现风险梯度但不至于过激
_DEFAULT_TOKEN_VOLS = [0.02, 0.08, 0.18]


def _gbm_path(
    rng: np.random.Generator,
    n: int,
    annual_vol: float,
    annual_drift: float = 0.0,
    dt: float = 1 / 365,
    start_price: float = 1.0,
) -> np.ndarray:
    """几何 Brownian 运动价格路径。

    S_t = S_{t-1} × exp((drift - 0.5·σ²)·dt + σ·√dt·Z)
    """
    if n <= 0:
        return np.array([start_price])
    z = rng.standard_normal(n)
    log_returns = (annual_drift - 0.5 * annual_vol ** 2) * dt + annual_vol * np.sqrt(dt) * z
    log_returns[0] = 0.0  # 起点价格 = start_price
    return start_price * np.exp(np.cumsum(log_returns))


def generate_sample_data(
    pool_ids: Optional[List[str]] = None,
    n_days: int = 365,
    seed: int = 42,
    apy_base: float = 0.05,
    apy_step_per_pool: float = 0.02,
    apy_drift_scale: float = 0.0015,
    crash_pool_index: int = 1,
    crash_window: Tuple[int, int] = (200, 205),
    crash_factor: float = 0.3,
    gas_spike_window: Tuple[int, int] = (150, 155),
    gas_spike_factor: float = 5.0,
    base_fee: float = 1e-7,
    priority_fee: float = 5e-8,
    base_tvl: float = 1e8,
    tvl_noise: float = 5e6,
    oracle_drift_scale: float = 0.003,
    token_vols: Optional[List[float]] = None,
    token_annual_drift: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """合成示例数据。返回 (pool_df, gas_df)。

    新增参数：
      token_vols：每池的年化波动率，长度需匹配 pool_ids；缺省按 _DEFAULT_TOKEN_VOLS
                  其中第一项作为稳定币代理（~2%），逐级升高
      token_annual_drift：基础年化漂移（默认 5%，给一点正向期望，避免过度悲观）
    """
    if pool_ids is None:
        pool_ids = list(_DEFAULT_POOL_IDS)
    if not pool_ids:
        raise ValueError("pool_ids 不能为空")
    if not (0 <= crash_pool_index < len(pool_ids)):
        raise ValueError(
            f"crash_pool_index={crash_pool_index} 越界（pool 数={len(pool_ids)}）"
        )

    n_pools = len(pool_ids)
    if token_vols is None:
        # 按 _DEFAULT_TOKEN_VOLS 长度循环复用
        token_vols = [_DEFAULT_TOKEN_VOLS[i % len(_DEFAULT_TOKEN_VOLS)]
                      for i in range(n_pools)]
    if len(token_vols) != n_pools:
        raise ValueError(
            f"token_vols 长度 {len(token_vols)} 与 pool_ids {n_pools} 不匹配"
        )

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_days, freq="D")

    pool_rows: list[dict] = []
    for i, pid in enumerate(pool_ids):
        # ---- APY 路径 ----
        base = apy_base + apy_step_per_pool * i
        apy_innovations = rng.normal(0, apy_drift_scale, n_days)
        apy_path = base + apy_innovations.cumsum() * 0.3
        # 不再 clip 到 0.001，让 crash 期间能短暂为负
        apy_path = np.clip(apy_path, -0.05, 1.0)
        # 闪崩注入：在 crash 池上覆盖一个更激烈的负面
        if i == crash_pool_index and crash_window is not None:
            lo, hi = min(crash_window[0], n_days), min(crash_window[1], n_days)
            if lo < hi:    # 仅当窗口落在数据范围内才注入
                apy_path[lo:hi] = apy_path[lo:hi] * crash_factor - 0.02
                # 后续也带一点尾巴的负面影响（恢复需 5 天）
                tail_lo, tail_hi = hi, min(hi + 5, n_days)
                if tail_lo < tail_hi:
                    apy_path[tail_lo:tail_hi] *= 0.6

        # ---- TVL 路径（轻噪声）----
        tvls = base_tvl + rng.normal(0, tvl_noise, n_days)

        # ---- token_price 路径（GBM）----
        token_path = _gbm_path(
            rng, n_days,
            annual_vol=float(token_vols[i]),
            annual_drift=token_annual_drift,
            start_price=1.0,
        )
        # crash 池在 crash 窗口内追加一次价格冲击 -15%
        if i == crash_pool_index and crash_window is not None:
            lo, hi = min(crash_window[0], n_days), min(crash_window[1], n_days)
            if lo < hi:
                shock = np.linspace(1.0, 0.85, hi - lo)
                token_path[lo:hi] *= shock
                # 恢复期：缓慢回升到 95%
                tail_lo, tail_hi = hi, min(hi + 10, n_days)
                if tail_lo < tail_hi:
                    recover = np.linspace(0.85, 0.95, tail_hi - tail_lo)
                    token_path[tail_lo:tail_hi] *= recover / (token_path[tail_lo:tail_hi] / token_path[lo])

        # ---- oracle_price（独立游走，提供 LVR）----
        oracle_walk = np.cumsum(rng.normal(0, oracle_drift_scale, n_days))
        oracle_path = token_path * (1.0 + oracle_walk * 0.1)

        for t, ts in enumerate(timestamps):
            pool_rows.append({
                "timestamp": ts,
                "pool_id": pid,
                "apy": float(apy_path[t]),
                "tvl": float(max(tvls[t], 1e6)),
                "token_price": float(token_path[t]),
                "oracle_price": float(oracle_path[t]),
            })
    pool_df = pd.DataFrame(pool_rows)

    gas_rows: list[dict] = []
    for t, ts in enumerate(timestamps):
        bf = base_fee
        pf = priority_fee
        if gas_spike_window is not None:
            lo, hi = gas_spike_window
            if lo <= t < hi:
                bf *= gas_spike_factor
                pf *= gas_spike_factor
        gas_rows.append({
            "timestamp": ts,
            "base_fee": bf,
            "priority_fee": pf,
        })
    gas_df = pd.DataFrame(gas_rows)

    return pool_df, gas_df
