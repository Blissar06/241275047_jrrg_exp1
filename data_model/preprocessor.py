"""数据清洗与 FR-02 收益拥挤衰减。

清洗操作：
  - align_timeseries：reindex 到统一频率
  - interpolate_missing：线性插值
  - remove_outliers_iqr：3*IQR 异常值剔除（前后值填充）

业务函数：
  - apply_capacity_decay：FR-02 资金容量影响下的有效 APY
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------- 时序对齐与缺失值处理 ----------

def align_timeseries(
    df: pd.DataFrame,
    freq: str = "D",
    timestamp_col: str = "timestamp",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """将时序 DataFrame reindex 到规则频率网格。

    若提供 group_col，则各组独立 reindex（避免不同池之间的时间网格相互污染）。
    """
    if group_col is None:
        s = df.set_index(timestamp_col).sort_index()
        if s.empty:
            return df.copy()
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq=freq)
        return s.reindex(full_idx).rename_axis(timestamp_col).reset_index()

    parts: List[pd.DataFrame] = []
    for key, g in df.groupby(group_col, sort=True):
        g = g.set_index(timestamp_col).sort_index()
        if g.empty:
            continue
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        g = g.reindex(full_idx).rename_axis(timestamp_col)
        g[group_col] = key
        parts.append(g.reset_index())
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def interpolate_missing(
    df: pd.DataFrame,
    numeric_cols: List[str],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """线性插值填充指定数值列的 NaN。

    若提供 group_col，每组独立插值，避免跨池数据污染。
    所有插值动作均写日志，不静默。
    """
    df = df.copy()
    if group_col is None:
        for col in numeric_cols:
            n_before = int(df[col].isna().sum())
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            n_after = int(df[col].isna().sum())
            if n_before > 0:
                logger.info(
                    "interpolated %d NaN in column=%s (residual=%d)",
                    n_before - n_after, col, n_after,
                )
        return df

    for key, g in df.groupby(group_col, sort=True):
        for col in numeric_cols:
            n_before = int(g[col].isna().sum())
            filled = g[col].interpolate(method="linear", limit_direction="both")
            df.loc[g.index, col] = filled
            n_after = int(filled.isna().sum())
            if n_before > 0:
                logger.info(
                    "interpolated %d NaN in column=%s group=%s=%s (residual=%d)",
                    n_before - n_after, col, group_col, key, n_after,
                )
    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    numeric_cols: List[str],
    k: float = 3.0,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """3*IQR 异常值剔除（用前后值填充）。

    剔除规则：x < Q1 - k*IQR 或 x > Q3 + k*IQR 视为异常 → 置 NaN → ffill/bfill 修复。
    所有剔除均写 WARN 日志，不静默忽略（满足 NFR-03）。
    """
    df = df.copy()

    def _clean_block(block: pd.DataFrame, scope_label: str) -> pd.DataFrame:
        for col in numeric_cols:
            q1 = block[col].quantile(0.25)
            q3 = block[col].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lo = q1 - k * iqr
            hi = q3 + k * iqr
            mask = (block[col] < lo) | (block[col] > hi)
            n_out = int(mask.sum())
            if n_out > 0:
                logger.warning(
                    "removed %d outliers in column=%s scope=%s bounds=[%.6f, %.6f]",
                    n_out, col, scope_label, lo, hi,
                )
                block.loc[mask, col] = np.nan
                block[col] = block[col].ffill().bfill()
        return block

    if group_col is None:
        return _clean_block(df, scope_label="<all>")

    for key, g in df.groupby(group_col, sort=True):
        cleaned = _clean_block(g.copy(), scope_label=f"{group_col}={key}")
        df.loc[g.index, numeric_cols] = cleaned[numeric_cols].values
    return df


# ---------- FR-02 收益拥挤衰减 ----------

def apply_capacity_decay(
    apy_nominal: Decimal,
    tvl: Decimal,
    capital: Decimal,
    pool_kind: str = "yield",
    utilization: Optional[Decimal] = None,
) -> Decimal:
    """FR-02 收益拥挤衰减：注入资金会稀释名义 APY。

    通用模型：
        APY_actual = APY_nominal × TVL / (TVL + Capital)
    含义：自有资金 Capital 注入池后，自身在池内的份额变为 Capital / (TVL + Capital)，
    池总收益按比例分摊，因此个体 APY 被稀释。

    针对借贷池（pool_kind='lending' 且提供 utilization）做二阶段修正：
      - U < kink (0.8)：在通用模型基础上再乘 (1 - U·0.1)，体现剩余可借空间
      - U >= kink：通用模型结果除以 (1 + 5·(U - kink))，体现利率曲线超线性上升导致
        新增资金边际收益骤降

    参数全部为 Decimal；返回 Decimal。
    """
    if not isinstance(apy_nominal, Decimal):
        apy_nominal = Decimal(str(apy_nominal))
    if not isinstance(tvl, Decimal):
        tvl = Decimal(str(tvl))
    if not isinstance(capital, Decimal):
        capital = Decimal(str(capital))

    denom = tvl + capital
    if denom <= 0:
        return Decimal(0)

    base = apy_nominal * tvl / denom

    if pool_kind == "lending" and utilization is not None:
        if not isinstance(utilization, Decimal):
            utilization = Decimal(str(utilization))
        kink = Decimal("0.8")
        if utilization < kink:
            return base * (Decimal(1) - utilization * Decimal("0.1"))
        penalty = Decimal(1) + (utilization - kink) * Decimal(5)
        return base / penalty

    return base
