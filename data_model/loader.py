"""CSV 数据加载与时序对齐。

输入：原始 pool / gas CSV
输出：List[AssetSnapshot]，按 tick 顺序排列，可直接喂给回测引擎。
"""
from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics

logger = logging.getLogger(__name__)

POOL_REQUIRED_COLUMNS = {"timestamp", "pool_id", "apy", "tvl", "token_price"}
GAS_REQUIRED_COLUMNS = {"timestamp", "base_fee", "priority_fee"}


def _to_decimal(x: Any) -> Decimal:
    """通过 str 中间格式安全转换为 Decimal，避免 float -> Decimal 的精度误差。"""
    if isinstance(x, Decimal):
        return x
    if pd.isna(x):
        return Decimal(0)
    return Decimal(str(x))


def load_pool_csv(path: str | Path) -> pd.DataFrame:
    """加载池时序 CSV。

    必需列：timestamp, pool_id, apy, tvl, token_price
    可选列：vol_30d, utilization, pool_kind
    """
    df = pd.read_csv(path)
    missing = POOL_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Pool CSV {path} 缺少必需列: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["pool_id", "timestamp"]).reset_index(drop=True)
    logger.info(
        "loaded pool csv: rows=%d pools=%d range=[%s, %s]",
        len(df),
        df["pool_id"].nunique(),
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    return df


def load_gas_csv(path: str | Path) -> pd.DataFrame:
    """加载 Gas 时序 CSV。必需列：timestamp, base_fee, priority_fee。"""
    df = pd.read_csv(path)
    missing = GAS_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Gas CSV {path} 缺少必需列: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "loaded gas csv: rows=%d range=[%s, %s]",
        len(df),
        df["timestamp"].min(),
        df["timestamp"].max(),
    )
    return df


def build_asset_snapshots(
    pool_df: pd.DataFrame,
    gas_df: pd.DataFrame,
    config: dict,
    momentum_window: Optional[int] = None,
) -> List[AssetSnapshot]:
    """对齐 pool / gas 时序后，构建 AssetSnapshot 列表。

    每个 snapshot 对应一个时间步：
      - pools 中每个 PoolMetrics 携带长度最多为 momentum_window 的回看 APY 序列
      - env 携带该 tick 的 base_fee / priority_fee / oracle_price
    """
    if momentum_window is None:
        momentum_window = int(config.get("momentum_window", 14))
    if momentum_window <= 0:
        raise ValueError(f"momentum_window 必须为正整数，got {momentum_window}")

    pool_ts = set(pool_df["timestamp"].unique())
    gas_ts = set(gas_df["timestamp"].unique())
    common = sorted(pool_ts & gas_ts)
    if not common:
        raise ValueError(
            "pool_df 与 gas_df 没有重叠的 timestamp —— 请先用 preprocessor.align_timeseries 对齐"
        )

    pool_ids = sorted(pool_df["pool_id"].unique())
    pool_indexed: Dict[str, pd.DataFrame] = {
        pid: pool_df[pool_df["pool_id"] == pid].set_index("timestamp").sort_index()
        for pid in pool_ids
    }
    gas_indexed = gas_df.set_index("timestamp").sort_index()

    has_vol = "vol_30d" in pool_df.columns
    has_oracle = "oracle_price" in pool_df.columns

    snapshots: List[AssetSnapshot] = []
    for tick_idx, ts in enumerate(common):
        pools: Dict[str, PoolMetrics] = {}
        oracle_price: Dict[str, Decimal] = {}

        gas_row = gas_indexed.loc[ts]
        base_fee = _to_decimal(gas_row["base_fee"])
        prio_fee = _to_decimal(gas_row["priority_fee"])

        for pid in pool_ids:
            pdf = pool_indexed[pid]
            if ts not in pdf.index:
                continue
            row = pdf.loc[ts]
            history = pdf.loc[pdf.index <= ts].tail(momentum_window)
            apy_tuple = tuple(_to_decimal(v) for v in history["apy"].values)

            tvl = _to_decimal(row["tvl"])
            token_price = _to_decimal(row["token_price"])
            vol_30d = _to_decimal(row["vol_30d"]) if has_vol else Decimal(0)

            # 同样窗口的 token_price 回看序列（用于价格风险评分）
            token_price_tuple = tuple(
                _to_decimal(v) for v in history["token_price"].values
            ) if "token_price" in history.columns else ()
            pools[pid] = PoolMetrics(
                pool_id=pid,
                apy_series=apy_tuple,
                tvl=tvl,
                vol_30d=vol_30d,
                token_price=token_price,
                gas_base_fee=base_fee,
                token_price_series=token_price_tuple,
            )
            # oracle_price 列若存在则用之，否则回退到 token_price（向后兼容）
            oracle_price[pid] = (
                _to_decimal(row["oracle_price"]) if has_oracle else token_price
            )

        ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        env = EnvSnapshot(
            tick=tick_idx,
            timestamp=ts_py,
            oracle_price=oracle_price,
            gas_base_fee=base_fee,
            gas_priority_fee=prio_fee,
        )
        snapshots.append(AssetSnapshot(tick=tick_idx, pools=pools, env=env))

    logger.info("built %d asset snapshots across %d pools", len(snapshots), len(pool_ids))
    return snapshots
