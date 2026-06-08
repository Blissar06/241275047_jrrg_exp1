"""量化绩效指标（FR-08）。

输入：
  - nav_log：BacktestEngine 输出的 DataFrame（必含 nav 列）
  - trade_log：必含 operation/gas_cost/slippage_cost/lvr_cost 列
  - reinvest_log：可选；若提供，复投 gas 计入 total_gas_cost

输出：MetricsReport（所有金融量为 Decimal，精度 ≥ 8 位）。

惯例：
  - periods_per_year 默认 365（DeFi 日级回测；股票场景可传 252）
  - 摩擦类指标仅汇总 operation == "ROTATE" 的实际成本，
    不汇总 HOLD 行（HOLD 的 gas/slippage 是评估时假设值，未真实发生）
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class MetricsReport:
    """绩效报表数据对象。所有比率字段无量纲。"""

    n_periods: int
    annualized_return: Decimal
    annualized_volatility: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    total_gas_cost: Decimal
    total_slippage_cost: Decimal
    total_lvr_cost: Decimal
    total_friction_cost: Decimal

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, Decimal) else v) for k, v in asdict(self).items()}


# =================================================================
# 工具函数
# =================================================================

def _to_decimal(x: float) -> Decimal:
    """通过 str 中转，避免 float→Decimal 精度漂移。"""
    return Decimal(str(float(x)))


def _daily_returns(nav_series: pd.Series) -> pd.Series:
    """nav_t / nav_{t-1} - 1；丢弃首项 NaN。"""
    return nav_series.pct_change().dropna()


# =================================================================
# 单项指标
# =================================================================

def annualized_return(nav_series: pd.Series, periods_per_year: int = 365) -> Decimal:
    """(NAV_end / NAV_start)^(periods_per_year / n_periods) - 1。"""
    if len(nav_series) < 2:
        return Decimal(0)
    start = float(nav_series.iloc[0])
    end = float(nav_series.iloc[-1])
    if start <= 0:
        return Decimal(0)
    n = len(nav_series) - 1
    return _to_decimal((end / start) ** (periods_per_year / n) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 365) -> Decimal:
    """日收益率标准差 × sqrt(periods_per_year)。

    阈值 1e-12 用来吃掉「恒定收益率序列」由于浮点累积产生的 ~1e-17 量级噪音 ——
    否则 sharpe = ar / av 会出现 1e29 级的伪值。
    """
    if len(returns) < 2:
        return Decimal(0)
    daily_std = float(returns.std(ddof=0))
    if daily_std < 1e-12:
        return Decimal(0)
    return _to_decimal(daily_std * (periods_per_year ** 0.5))


def max_drawdown(nav_series: pd.Series) -> Decimal:
    """1 - nav / nav.cummax()，取最大值。"""
    if nav_series.empty:
        return Decimal(0)
    arr = np.asarray(nav_series.values, dtype=float)
    peaks = np.maximum.accumulate(arr)
    safe = np.where(peaks > 0, peaks, 1.0)
    drawdowns = np.where(peaks > 0, 1.0 - arr / safe, 0.0)
    return _to_decimal(float(np.max(drawdowns)))


def sharpe_ratio(
    nav_series: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 365,
) -> Decimal:
    """(年化收益 - rf) / 年化波动率。波动率为 0 时返回 0。"""
    returns = _daily_returns(nav_series)
    if returns.empty:
        return Decimal(0)
    av = annualized_volatility(returns, periods_per_year)
    if av == 0:
        return Decimal(0)
    ar = annualized_return(nav_series, periods_per_year)
    return (ar - _to_decimal(risk_free)) / av


def sortino_ratio(
    nav_series: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = 365,
) -> Decimal:
    """(年化收益 - rf) / 下行年化波动率（仅负收益的 std × sqrt(periods))。"""
    returns = _daily_returns(nav_series)
    if returns.empty:
        return Decimal(0)
    downside = returns[returns < 0]
    if downside.empty:
        return Decimal(0)
    daily_dvol = float(downside.std(ddof=0))
    if daily_dvol == 0.0:
        return Decimal(0)
    annual_dvol = _to_decimal(daily_dvol * (periods_per_year ** 0.5))
    if annual_dvol == 0:
        return Decimal(0)
    ar = annualized_return(nav_series, periods_per_year)
    return (ar - _to_decimal(risk_free)) / annual_dvol


def calmar_ratio(nav_series: pd.Series, periods_per_year: int = 365) -> Decimal:
    """年化收益 / 最大回撤。MDD=0 时返回 0。"""
    mdd = max_drawdown(nav_series)
    if mdd == 0:
        return Decimal(0)
    ar = annualized_return(nav_series, periods_per_year)
    return ar / mdd


# =================================================================
# 摩擦成本汇总
# =================================================================

def _sum_decimal(series: pd.Series) -> Decimal:
    return _to_decimal(float(series.sum())) if len(series) > 0 else Decimal(0)


def total_gas_cost(
    trade_log: pd.DataFrame,
    reinvest_log: Optional[pd.DataFrame] = None,
) -> Decimal:
    """实际发生的 gas 总开销。

    包括：ROTATE 行的 gas_cost + reinvest_log 的 gas_cost
    （HOLD 行的 gas_cost 是评估期假设值，不计入）。
    """
    gas = Decimal(0)
    if not trade_log.empty and "operation" in trade_log.columns:
        rotated = trade_log[trade_log["operation"] == "ROTATE"]
        if not rotated.empty:
            gas += _sum_decimal(rotated["gas_cost"])
    if reinvest_log is not None and not reinvest_log.empty:
        gas += _sum_decimal(reinvest_log["gas_cost"])
    return gas


def total_slippage(trade_log: pd.DataFrame) -> Decimal:
    if trade_log.empty or "operation" not in trade_log.columns:
        return Decimal(0)
    rotated = trade_log[trade_log["operation"] == "ROTATE"]
    return _sum_decimal(rotated["slippage_cost"]) if not rotated.empty else Decimal(0)


def total_lvr(trade_log: pd.DataFrame) -> Decimal:
    if trade_log.empty or "operation" not in trade_log.columns:
        return Decimal(0)
    rotated = trade_log[trade_log["operation"] == "ROTATE"]
    return _sum_decimal(rotated["lvr_cost"]) if not rotated.empty else Decimal(0)


# =================================================================
# 一次性聚合
# =================================================================

def compute_metrics(
    nav_log: pd.DataFrame,
    trade_log: pd.DataFrame,
    reinvest_log: Optional[pd.DataFrame] = None,
    risk_free: float = 0.0,
    periods_per_year: int = 365,
) -> MetricsReport:
    """打包所有指标为 MetricsReport。"""
    if nav_log.empty or "nav" not in nav_log.columns:
        return MetricsReport(
            n_periods=0,
            annualized_return=Decimal(0),
            annualized_volatility=Decimal(0),
            max_drawdown=Decimal(0),
            sharpe_ratio=Decimal(0),
            sortino_ratio=Decimal(0),
            calmar_ratio=Decimal(0),
            total_gas_cost=Decimal(0),
            total_slippage_cost=Decimal(0),
            total_lvr_cost=Decimal(0),
            total_friction_cost=Decimal(0),
        )

    nav = nav_log["nav"]
    rets = _daily_returns(nav)

    gas = total_gas_cost(trade_log, reinvest_log)
    slip = total_slippage(trade_log)
    lvr = total_lvr(trade_log)

    return MetricsReport(
        n_periods=len(nav),
        annualized_return=annualized_return(nav, periods_per_year),
        annualized_volatility=annualized_volatility(rets, periods_per_year),
        max_drawdown=max_drawdown(nav),
        sharpe_ratio=sharpe_ratio(nav, risk_free, periods_per_year),
        sortino_ratio=sortino_ratio(nav, risk_free, periods_per_year),
        calmar_ratio=calmar_ratio(nav, periods_per_year),
        total_gas_cost=gas,
        total_slippage_cost=slip,
        total_lvr_cost=lvr,
        total_friction_cost=gas + slip + lvr,
    )
