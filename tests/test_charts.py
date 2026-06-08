"""Phase 7-A：图表工厂模块单元测试。

冒烟级别 —— 验证函数在典型输入下返回有效 Plotly Figure 而不抛错；
不做视觉断言，那是 UI 自身职责。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from report.attribution import AttributionReport
from ui.charts import (
    apy_heatmap,
    apy_history,
    attribution_radar_multi,
    cost_composition_stacked,
    drawdown_underwater,
    gas_timeline,
    nav_with_trade_markers,
    position_timeline,
    rolling_sharpe,
    tvl_history,
)


# ---------- fixtures ----------

@pytest.fixture
def nav_log():
    n = 60
    return pd.DataFrame({
        "tick": range(n),
        "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)],
        "nav": [100_000 + i * 100 for i in range(n)],
        "principal": [100_000 + i * 100 for i in range(n)],
        "pending_reward": [0.0] * n,
        "cash": [0.0] * n,
        "pool_id": ["pool_A"] * 20 + ["pool_B"] * 20 + ["pool_C"] * 20,
        "env_gas_base_fee": [1e-7] * 30 + [5e-7] * 5 + [1e-7] * 25,
        "env_gas_priority_fee": [5e-8] * 60,
    })


@pytest.fixture
def trade_log():
    return pd.DataFrame([
        {"tick": 0, "timestamp": datetime(2024, 1, 1), "operation": "ROTATE",
         "from_pool_id": None, "to_pool_id": "pool_A",
         "amount": 100_000.0, "gas_cost": 10.0, "slippage_cost": 5.0,
         "lvr_cost": 1.0, "expected_gain": 50.0, "decision_reason": "OK"},
        {"tick": 20, "timestamp": datetime(2024, 1, 21), "operation": "ROTATE",
         "from_pool_id": "pool_A", "to_pool_id": "pool_B",
         "amount": 102_000.0, "gas_cost": 12.0, "slippage_cost": 6.0,
         "lvr_cost": 1.5, "expected_gain": 60.0, "decision_reason": "OK"},
        {"tick": 40, "timestamp": datetime(2024, 2, 10), "operation": "HOLD",
         "from_pool_id": "pool_B", "to_pool_id": "pool_B",
         "amount": 0.0, "gas_cost": 0.0, "slippage_cost": 0.0,
         "lvr_cost": 0.0, "expected_gain": 0.0, "decision_reason": "SAME_POOL"},
    ])


@pytest.fixture
def pool_df():
    rows = []
    for pid in ["pool_A", "pool_B", "pool_C"]:
        base = {"pool_A": 0.05, "pool_B": 0.07, "pool_C": 0.09}[pid]
        for t in range(60):
            rows.append({
                "timestamp": datetime(2024, 1, 1) + timedelta(days=t),
                "pool_id": pid,
                "apy": base + 0.001 * t,
                "tvl": 10_000_000 + 100_000 * t,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def gas_df():
    return pd.DataFrame([
        {"timestamp": datetime(2024, 1, 1) + timedelta(days=t),
         "base_fee": 1e-7 * (5.0 if 30 <= t < 35 else 1.0),
         "priority_fee": 5e-8}
        for t in range(60)
    ])


# ---------- 1. NAV + 标记 ----------

def test_nav_with_trade_markers_returns_figure(nav_log, trade_log):
    fig = nav_with_trade_markers(nav_log, trade_log)
    assert isinstance(fig, go.Figure)
    trace_names = {t.name for t in fig.data}
    assert "实际 NAV" in trace_names
    assert "ROTATE" in trace_names


def test_nav_with_theoretical_overlay(nav_log, trade_log):
    theo = pd.Series([100_000 + i * 150 for i in range(60)])
    fig = nav_with_trade_markers(nav_log, trade_log, theoretical_nav=theo)
    trace_names = {t.name for t in fig.data}
    assert "理论最优（无摩擦）" in trace_names


def test_nav_empty_input_returns_placeholder():
    fig = nav_with_trade_markers(pd.DataFrame(), pd.DataFrame())
    assert isinstance(fig, go.Figure)


# ---------- 2/3. APY / TVL ----------

def test_apy_history_returns_one_trace_per_pool(pool_df):
    fig = apy_history(pool_df)
    assert len(fig.data) == 3


def test_tvl_history_returns_one_trace_per_pool(pool_df):
    fig = tvl_history(pool_df)
    assert len(fig.data) == 3


def test_apy_history_empty_returns_placeholder():
    fig = apy_history(pd.DataFrame())
    assert isinstance(fig, go.Figure)


# ---------- 4. Gas timeline ----------

def test_gas_timeline_prefers_nav_log_env_columns(nav_log, gas_df):
    fig = gas_timeline(gas_df, nav_log=nav_log)
    assert len(fig.data) == 2


def test_gas_timeline_falls_back_to_gas_df(gas_df):
    fig = gas_timeline(gas_df)
    assert len(fig.data) == 2


def test_gas_timeline_empty_returns_placeholder():
    fig = gas_timeline(None)
    assert isinstance(fig, go.Figure)


# ---------- 5. Drawdown ----------

def test_drawdown_underwater_returns_figure(nav_log):
    fig = drawdown_underwater(nav_log)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_drawdown_underwater_monotone_increasing_nav_is_zero():
    df = pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)],
        "nav": [100 + i for i in range(10)],
    })
    fig = drawdown_underwater(df)
    y = fig.data[0].y
    assert max(y) <= 0  # 单调升 → 回撤 <= 0


# ---------- 6. Position timeline ----------

def test_position_timeline_creates_segments_per_pool_change(nav_log):
    fig = position_timeline(nav_log)
    # nav_log 有 3 段：A, B, C
    assert len(fig.data) == 3


def test_position_timeline_treats_none_as_cash():
    df = pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)],
        "pool_id": [None, None, "pool_A", "pool_A", "pool_A"],
    })
    fig = position_timeline(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # 现金 段 + pool_A 段


# ---------- 7. APY heatmap ----------

def test_apy_heatmap_shape_matches_pivot(pool_df):
    fig = apy_heatmap(pool_df)
    assert isinstance(fig, go.Figure)
    z = fig.data[0].z
    assert z.shape == (3, 60)   # 3 池 × 60 天


# ---------- 8. Rolling Sharpe ----------

def test_rolling_sharpe_returns_figure(nav_log):
    fig = rolling_sharpe(nav_log, window=10)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_rolling_sharpe_too_short_returns_placeholder():
    df = pd.DataFrame({
        "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)],
        "nav": [100, 101, 102, 103, 104],
    })
    fig = rolling_sharpe(df, window=30)
    assert isinstance(fig, go.Figure)


# ---------- 9. Cost composition ----------

def test_cost_composition_stacks_three_components(trade_log):
    fig = cost_composition_stacked(trade_log)
    assert len(fig.data) == 3
    names = {t.name for t in fig.data}
    assert names == {"Gas", "Slippage", "LVR"}


def test_cost_composition_no_rotates_returns_placeholder():
    only_holds = pd.DataFrame([
        {"tick": 0, "timestamp": datetime(2024, 1, 1), "operation": "HOLD",
         "gas_cost": 0.0, "slippage_cost": 0.0, "lvr_cost": 0.0},
    ])
    fig = cost_composition_stacked(only_holds)
    assert isinstance(fig, go.Figure)


# ---------- 归因雷达多策略 ----------

def _fake_attribution(gas_pct: float) -> AttributionReport:
    z = Decimal(0)
    return AttributionReport(
        theoretical_total_return=Decimal("1000"),
        actual_return=Decimal("900"),
        rotation_count=1, reinvest_count=1,
        gas_cost=z, slippage_cost=z, lvr_cost=z, rotation_idle_cost=z,
        gas_cost_pct=Decimal(str(gas_pct)),
        slippage_pct=Decimal("2"), lvr_pct=Decimal("1"),
        rotation_idle_pct=Decimal("0.5"),
        total_friction_pct=Decimal("5"),
    )


def test_attribution_radar_multi_renders_each_strategy():
    """3 个策略 → 3 个雷达 trace + 1 个 bar (idle) = 4 个 trace。"""
    strategies = {
        "A": _fake_attribution(1.0),
        "B": _fake_attribution(3.0),
        "C": _fake_attribution(5.0),
    }
    fig = attribution_radar_multi(strategies)
    # 拆 subplot 后：3 个 Scatterpolar + 1 个 Bar
    assert len(fig.data) == 4
    polar_traces = [t for t in fig.data if t.type == "scatterpolar"]
    assert len(polar_traces) == 3


def test_attribution_radar_multi_empty_returns_placeholder():
    fig = attribution_radar_multi({})
    assert isinstance(fig, go.Figure)
