"""收益归因分析（FR-08）。

把「理论最优 - 实际净收益」这个差额拆解为四类来源：

  total_gap = theoretical_return - actual_return
            = gas_cost + slippage_cost + lvr_cost + rotation_idle

其中：
  - theoretical_return：每 tick 都持有 max-APY 池、零摩擦的复利净收益
  - gas / slippage / lvr：实际发生的链上摩擦成本（含复投 gas）
  - rotation_idle：调仓滞后的折损（位于次优池而非 max-APY 池的累计代价）

百分比口径：以 theoretical_total_return 为分母（绝对收益基准）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

from report.metrics import (
    _to_decimal,
    total_gas_cost,
    total_lvr,
    total_slippage,
)


@dataclass(frozen=True, slots=True)
class AttributionReport:
    """收益归因数据对象。所有 *_pct 字段为百分比（%）单位。"""

    theoretical_total_return: Decimal     # 无摩擦理论净收益
    actual_return: Decimal                # 实际净收益
    rotation_count: int                   # 总轮动次数
    reinvest_count: int                   # 总复投次数
    gas_cost: Decimal
    slippage_cost: Decimal
    lvr_cost: Decimal
    rotation_idle_cost: Decimal           # 调仓空窗折损（绝对值）
    gas_cost_pct: Decimal
    slippage_pct: Decimal
    lvr_pct: Decimal
    rotation_idle_pct: Decimal
    total_friction_pct: Decimal           # gas + slippage + lvr 总占比

    def to_dict(self) -> dict:
        return {
            k: (float(v) if isinstance(v, Decimal) else v)
            for k, v in asdict(self).items()
        }

    def export_csv(self, path: str | Path) -> None:
        df = pd.DataFrame([self.to_dict()])
        df.to_csv(path, index=False)


# =================================================================
# 理论最优路径
# =================================================================

def theoretical_nav_path(
    score_log: pd.DataFrame,
    initial_capital: Decimal,
    periods_per_year: int = 365,
) -> pd.Series:
    """每 tick 持有 max-APY 池的零摩擦复利路径。

    返回与 nav_log 等长的 Series（按 tick 顺序）。
    score_log 必须含 tick / apy 列。
    """
    if score_log.empty or "apy" not in score_log.columns:
        return pd.Series([float(initial_capital)], dtype=float)

    grouped = score_log.groupby("tick")["apy"].max().sort_index()
    nav = Decimal(str(initial_capital))
    factor = Decimal(periods_per_year)

    nav_path: list[float] = []
    for apy in grouped.values:
        per_tick = _to_decimal(apy) / factor
        nav = nav * (Decimal(1) + per_tick)
        nav_path.append(float(nav))
    return pd.Series(nav_path, dtype=float)


# =================================================================
# 主入口
# =================================================================

def compute_attribution(
    nav_log: pd.DataFrame,
    trade_log: pd.DataFrame,
    score_log: pd.DataFrame,
    initial_capital: Decimal,
    reinvest_log: Optional[pd.DataFrame] = None,
    periods_per_year: int = 365,
) -> AttributionReport:
    """构造 AttributionReport。"""
    if nav_log.empty or "nav" not in nav_log.columns:
        return _empty_report()

    final_nav = _to_decimal(float(nav_log["nav"].iloc[-1]))
    actual_return = final_nav - Decimal(str(initial_capital))

    theo_path = theoretical_nav_path(score_log, Decimal(str(initial_capital)), periods_per_year)
    theo_return = _to_decimal(float(theo_path.iloc[-1])) - Decimal(str(initial_capital))

    # 摩擦成本（实际发生）
    gas = total_gas_cost(trade_log, reinvest_log)
    slip = total_slippage(trade_log)
    lvr = total_lvr(trade_log)
    friction_total = gas + slip + lvr

    # 调仓空窗折损：理论缺口 - 摩擦
    total_gap = theo_return - actual_return
    idle = total_gap - friction_total
    if idle < 0:
        # 数值噪音；理论上不应出现，但实际中浮点累积或外部数据可能让它略小于 0
        idle = Decimal(0)

    # 计数
    if not trade_log.empty and "operation" in trade_log.columns:
        rot_count = int((trade_log["operation"] == "ROTATE").sum())
    else:
        rot_count = 0
    rein_count = int(len(reinvest_log)) if reinvest_log is not None else 0

    # 百分比
    if theo_return > 0:
        denom = theo_return
        gas_pct = gas / denom * Decimal(100)
        slip_pct = slip / denom * Decimal(100)
        lvr_pct = lvr / denom * Decimal(100)
        idle_pct = idle / denom * Decimal(100)
        friction_pct = friction_total / denom * Decimal(100)
    else:
        gas_pct = slip_pct = lvr_pct = idle_pct = friction_pct = Decimal(0)

    return AttributionReport(
        theoretical_total_return=theo_return,
        actual_return=actual_return,
        rotation_count=rot_count,
        reinvest_count=rein_count,
        gas_cost=gas,
        slippage_cost=slip,
        lvr_cost=lvr,
        rotation_idle_cost=idle,
        gas_cost_pct=gas_pct,
        slippage_pct=slip_pct,
        lvr_pct=lvr_pct,
        rotation_idle_pct=idle_pct,
        total_friction_pct=friction_pct,
    )


def _empty_report() -> AttributionReport:
    z = Decimal(0)
    return AttributionReport(
        theoretical_total_return=z,
        actual_return=z,
        rotation_count=0,
        reinvest_count=0,
        gas_cost=z, slippage_cost=z, lvr_cost=z, rotation_idle_cost=z,
        gas_cost_pct=z, slippage_pct=z, lvr_pct=z,
        rotation_idle_pct=z, total_friction_pct=z,
    )
