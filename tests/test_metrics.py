"""Phase 4 命令 4-1：metrics 单元测试。"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from report.metrics import (
    MetricsReport,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    total_gas_cost,
    total_lvr,
    total_slippage,
)


def _nav_series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


# ========== annualized_return ==========

def test_annualized_return_doubling_in_one_year():
    """nav 由 100 涨到 200 共 365 个 tick，年化应为 100%。"""
    nav = _nav_series([100 + (100 * i / 365) for i in range(366)])
    ar = annualized_return(nav, periods_per_year=365)
    # 用 (200/100)^(365/365) - 1 = 1.0
    assert ar == pytest.approx(Decimal(1), rel=Decimal("0.01"))


def test_annualized_return_handles_short_series():
    assert annualized_return(_nav_series([100]), 365) == Decimal(0)
    assert annualized_return(_nav_series([]), 365) == Decimal(0)


def test_annualized_return_zero_start_returns_zero():
    assert annualized_return(_nav_series([0, 100]), 365) == Decimal(0)


# ========== max_drawdown ==========

def test_max_drawdown_simple_case():
    # 100 → 200 → 50 → 100；峰值 200，最低 50，MDD = 1 - 50/200 = 0.75
    nav = _nav_series([100, 200, 50, 100])
    assert max_drawdown(nav) == Decimal("0.75")


def test_max_drawdown_monotone_increasing_is_zero():
    nav = _nav_series([100, 110, 120, 130])
    assert max_drawdown(nav) == Decimal(0)


def test_max_drawdown_empty_returns_zero():
    assert max_drawdown(_nav_series([])) == Decimal(0)


# ========== volatility / Sharpe / Sortino ==========

def test_annualized_volatility_recovers_known_value():
    # 收益率 [0.01, -0.01, 0.01, -0.01]，std=0.01，年化 = 0.01 * sqrt(365) ≈ 0.191
    rets = pd.Series([0.01, -0.01, 0.01, -0.01])
    av = annualized_volatility(rets, 365)
    assert float(av) == pytest.approx(0.01 * (365 ** 0.5), abs=1e-6)


def test_sharpe_zero_when_no_volatility():
    nav = _nav_series([100, 110, 121, 133.1])  # 每天涨 10%，std=0
    assert sharpe_ratio(nav, risk_free=0.0, periods_per_year=365) == Decimal(0)


def test_sharpe_positive_for_up_trend_with_some_volatility():
    rng = np.random.default_rng(42)
    rets = rng.normal(loc=0.001, scale=0.005, size=200)
    nav_vals = [100.0]
    for r in rets:
        nav_vals.append(nav_vals[-1] * (1 + r))
    nav = _nav_series(nav_vals)
    sr = sharpe_ratio(nav, periods_per_year=365)
    assert sr > Decimal(0)


def test_sortino_zero_when_no_downside():
    # 全部正收益
    nav = _nav_series([100, 105, 110, 115])
    assert sortino_ratio(nav) == Decimal(0)


def test_sortino_distinct_from_sharpe_with_asymmetric_returns():
    # 有较大正收益 + 较小负收益 → Sortino 应明显高于 Sharpe
    rets = [0.05, -0.005, 0.04, -0.005, 0.05, -0.005, 0.04]
    nav_vals = [100.0]
    for r in rets:
        nav_vals.append(nav_vals[-1] * (1 + r))
    nav = _nav_series(nav_vals)
    sr = sharpe_ratio(nav)
    so = sortino_ratio(nav)
    assert so > sr


# ========== calmar ==========

def test_calmar_zero_when_no_drawdown():
    nav = _nav_series([100, 110, 120])
    assert calmar_ratio(nav) == Decimal(0)


def test_calmar_equals_ar_over_mdd():
    nav = _nav_series([100, 110, 90, 95])
    cm = calmar_ratio(nav)
    ar = annualized_return(nav)
    mdd = max_drawdown(nav)
    assert cm == ar / mdd


# ========== 摩擦成本汇总 ==========

def _make_trade_log(
    rotates: list[tuple[float, float, float]],   # (gas, slip, lvr)
    holds: int = 0,
) -> pd.DataFrame:
    rows = []
    t = 0
    for gas, slip, lvr in rotates:
        rows.append({
            "tick": t, "operation": "ROTATE",
            "gas_cost": gas, "slippage_cost": slip, "lvr_cost": lvr,
        })
        t += 1
    for _ in range(holds):
        rows.append({
            "tick": t, "operation": "HOLD",
            "gas_cost": 999.0, "slippage_cost": 999.0, "lvr_cost": 999.0,
        })
        t += 1
    return pd.DataFrame(rows)


def test_total_gas_excludes_hold_rows():
    df = _make_trade_log([(10, 1, 0), (20, 2, 0)], holds=5)
    assert total_gas_cost(df) == Decimal("30")


def test_total_gas_includes_reinvest_log():
    df = _make_trade_log([(10, 0, 0)])
    rein = pd.DataFrame([{"gas_cost": 5}, {"gas_cost": 3}])
    assert total_gas_cost(df, rein) == Decimal("18")


def test_total_slippage_lvr_only_from_rotate():
    df = _make_trade_log([(10, 5, 1), (10, 7, 2)], holds=3)
    assert total_slippage(df) == Decimal("12")
    assert total_lvr(df) == Decimal("3")


def test_total_costs_empty_log_returns_zero():
    empty = pd.DataFrame()
    assert total_gas_cost(empty) == Decimal(0)
    assert total_slippage(empty) == Decimal(0)
    assert total_lvr(empty) == Decimal(0)


# ========== compute_metrics 一站式 ==========

def test_compute_metrics_full_report_shape():
    nav = _nav_series([100 + i for i in range(365)])
    nav_log = pd.DataFrame({
        "tick": range(365),
        "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(365)],
        "nav": nav.values,
    })
    trade = _make_trade_log([(10, 5, 1)], holds=3)
    rep = compute_metrics(nav_log, trade)

    assert isinstance(rep, MetricsReport)
    assert rep.n_periods == 365
    assert rep.annualized_return > Decimal(0)
    assert rep.max_drawdown == Decimal(0)
    assert rep.total_gas_cost == Decimal("10")
    assert rep.total_friction_cost == Decimal("16")


def test_compute_metrics_to_dict_serializes_decimal():
    nav_log = pd.DataFrame({"nav": [100, 110, 120]})
    trade = pd.DataFrame()
    rep = compute_metrics(nav_log, trade)
    d = rep.to_dict()
    # 所有 Decimal 已转 float / int
    assert isinstance(d["annualized_return"], float)
    assert isinstance(d["n_periods"], int)


def test_compute_metrics_empty_nav_returns_zeros():
    rep = compute_metrics(pd.DataFrame(), pd.DataFrame())
    assert rep.n_periods == 0
    assert rep.annualized_return == Decimal(0)
    assert rep.sharpe_ratio == Decimal(0)


def test_metrics_decimal_precision_at_least_8_digits():
    """财务字段应保留 ≥ 8 位精度。"""
    nav = _nav_series([100.0 + (i * 0.0000001) for i in range(100)])
    ar = annualized_return(nav, 365)
    s = str(ar)
    # 既然 nav 微小变动，年化收益的有效位应反映到小数后 8 位
    if "." in s:
        decimals = s.split(".")[1]
        assert len(decimals) >= 8
