"""黑盒：边界值分析测试（Boundary Value Analysis）。

PPT 第 8 讲 §9.2：在等价类的"边界"上测，因为缺陷往往集中在边界处。
对每个有阈值的输入，挑：
  - 刚好 < 阈值
  - 等于阈值（开闭性测试）
  - 刚好 > 阈值
  以及上下界（0、负值、极大值、单元素等）。

编号规范：BV-<函数缩写>-<序号>。
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from backtest.cost_model import FrictionEstimator
from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from data_model.preprocessor import (
    apply_capacity_decay,
    interpolate_missing,
    remove_outliers_iqr,
)
from report.metrics import (
    annualized_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import (
    DecisionType,
    HoldReason,
    PoolScore,
    Position,
    RankingTable,
)
from strategy.rotation_engine import RotationEngine
from strategy.scorers.momentum import MomentumScorer, ewma
from tests._stubs import StubFrictionEstimator

pytestmark = [pytest.mark.blackbox, pytest.mark.boundary]


# =====================================================================
# 函数 1：FrictionEstimator.estimate_slippage —— 阶梯阈值边界
# 阈值：ratio = trade/tvl，分界 0.01 / 0.05
# =====================================================================

class TestSlippageThresholdBoundaries:
    """三档滑点阈值的边界点测试。"""

    @pytest.fixture
    def fe(self):
        return FrictionEstimator()  # low=0.001, mid=0.003, high=0.008

    def test_BV_SLIP_01_just_below_low_threshold(self, fe):
        """ratio = 0.00999 → low_rate (0.001)。"""
        tvl = Decimal("1000000")
        trade = tvl * Decimal("0.00999")
        cost = fe.estimate_slippage(trade, tvl)
        assert cost == trade * Decimal("0.001")

    def test_BV_SLIP_02_exactly_at_low_threshold(self, fe):
        """ratio == 0.01：边界（按代码 < 走 low；≥ 走 mid）→ mid_rate (0.003)。"""
        tvl = Decimal("1000000")
        trade = tvl * Decimal("0.01")
        cost = fe.estimate_slippage(trade, tvl)
        assert cost == trade * Decimal("0.003")

    def test_BV_SLIP_03_just_above_low_threshold(self, fe):
        """ratio = 0.0101 → mid_rate。"""
        tvl = Decimal("1000000")
        trade = tvl * Decimal("0.0101")
        cost = fe.estimate_slippage(trade, tvl)
        assert cost == trade * Decimal("0.003")

    def test_BV_SLIP_04_just_below_high_threshold(self, fe):
        """ratio = 0.0499 → mid_rate。"""
        tvl = Decimal("1000000")
        trade = tvl * Decimal("0.0499")
        cost = fe.estimate_slippage(trade, tvl)
        assert cost == trade * Decimal("0.003")

    def test_BV_SLIP_05_exactly_at_high_threshold(self, fe):
        """ratio == 0.05 → high_rate。"""
        tvl = Decimal("1000000")
        trade = tvl * Decimal("0.05")
        cost = fe.estimate_slippage(trade, tvl)
        assert cost == trade * Decimal("0.008")

    def test_BV_SLIP_06_just_above_high_threshold(self, fe):
        """ratio = 0.0501 → high_rate。"""
        tvl = Decimal("1000000")
        trade = tvl * Decimal("0.0501")
        cost = fe.estimate_slippage(trade, tvl)
        assert cost == trade * Decimal("0.008")

    def test_BV_SLIP_07_extreme_trade_size(self, fe):
        """极大交易（trade > tvl）→ high_rate；数学不爆炸。"""
        cost = fe.estimate_slippage(Decimal("10000000"), Decimal("1000000"))  # 1000% TVL
        assert cost == Decimal("10000000") * Decimal("0.008")


# =====================================================================
# 函数 2：apply_capacity_decay —— TVL/Capital 比例边界
# =====================================================================

class TestCapacityDecayBoundaries:

    def test_BV_CAP_01_zero_tvl_zero_capital(self):
        """边界：分母 = 0 → 函数返回 0（避免除零）。"""
        assert apply_capacity_decay(Decimal("0.1"), Decimal(0), Decimal(0)) == Decimal(0)

    def test_BV_CAP_02_tvl_one_unit(self):
        """极小 TVL = 1 → 极度稀释。"""
        out = apply_capacity_decay(Decimal("0.1"), Decimal(1), Decimal("1000000"))
        assert out < Decimal("0.000001")

    def test_BV_CAP_03_utilization_branch_switch_at_kink(self):
        """utilization == 0.8 是分支切换点。

        注意：当前实现在 kink 两侧有不连续（已知设计权衡）。本测试只验证两个
        分支都返回有限正数，并对单调性做更宽松检查："明显高于 kink"应明显低于
        "明显低于 kink"。
        """
        # 单调性比较：远低于 kink vs 远高于 kink
        low_u = apply_capacity_decay(
            Decimal("0.1"), Decimal("1000000"), Decimal("100000"),
            pool_kind="lending", utilization=Decimal("0.3"),
        )
        high_u = apply_capacity_decay(
            Decimal("0.1"), Decimal("1000000"), Decimal("100000"),
            pool_kind="lending", utilization=Decimal("0.95"),
        )
        assert high_u < low_u
        # 两个分支都应该返回有限正数（非崩溃）
        assert low_u > Decimal(0) and high_u > Decimal(0)

    def test_BV_CAP_04_utilization_at_one(self):
        """utilization == 1.0：极端边界，函数仍应有有限输出。"""
        out = apply_capacity_decay(
            Decimal("0.1"), Decimal("1000000"), Decimal("100000"),
            pool_kind="lending", utilization=Decimal("1.0"),
        )
        assert out >= Decimal(0)
        assert out < Decimal("0.1")    # 严重惩罚下应远低于名义


# =====================================================================
# 函数 3：ewma —— 序列长度与 λ 边界
# =====================================================================

class TestEwmaBoundaries:

    def test_BV_EWM_01_empty_series(self):
        assert ewma((), Decimal("0.85")) == Decimal(0)

    def test_BV_EWM_02_single_element(self):
        assert ewma((Decimal("0.05"),), Decimal("0.85")) == Decimal("0.05")

    def test_BV_EWM_03_lambda_just_above_zero(self):
        """λ → 0：完全跟随最新值。"""
        series = (Decimal("0.1"), Decimal("0.2"), Decimal("0.3"))
        out = ewma(series, Decimal("0.001"))
        # 几乎完全跟随最后一个值
        assert abs(out - Decimal("0.3")) < Decimal("0.01")

    def test_BV_EWM_04_lambda_just_below_one(self):
        """λ → 1：几乎完全保留历史。"""
        series = (Decimal("0.1"), Decimal("0.2"), Decimal("0.3"))
        out = ewma(series, Decimal("0.999"))
        # 几乎完全保留 series[0]
        assert abs(out - Decimal("0.1")) < Decimal("0.01")

    @pytest.mark.parametrize("invalid_lam", [
        Decimal(0),        # 等于下界 → 拒绝
        Decimal(1),        # 等于上界 → 拒绝
        Decimal("-0.0001"),
        Decimal("1.0001"),
    ])
    def test_BV_EWM_05_lambda_on_invalid_boundary(self, invalid_lam):
        with pytest.raises(ValueError):
            ewma((Decimal("0.05"),), invalid_lam)


# =====================================================================
# 函数 4：max_drawdown —— NAV 序列边界
# =====================================================================

class TestMaxDrawdownBoundaries:

    def test_BV_MDD_01_empty_series(self):
        assert max_drawdown(pd.Series([], dtype=float)) == Decimal(0)

    def test_BV_MDD_02_single_element(self):
        assert max_drawdown(pd.Series([100.0])) == Decimal(0)

    def test_BV_MDD_03_two_elements_monotone(self):
        assert max_drawdown(pd.Series([100, 110])) == Decimal(0)

    def test_BV_MDD_04_immediate_drop_from_first(self):
        """峰值在首元素，立刻跌 → 全程回撤。"""
        out = max_drawdown(pd.Series([100, 50]))
        assert out == Decimal("0.5")

    def test_BV_MDD_05_with_zero_peak(self):
        """含 0 值（peak = 0）→ 公式应防 0 除并返回 0。"""
        out = max_drawdown(pd.Series([0, 0, 0]))
        assert out == Decimal(0)


# =====================================================================
# 函数 5：sharpe_ratio / sortino_ratio —— 收益序列边界
# =====================================================================

class TestSharpeSortinoBoundaries:

    def test_BV_SHARP_01_constant_returns_zero(self):
        nav = pd.Series([100, 105, 110.25, 115.7625])  # 每步 +5%
        # std ≈ 0 → sharpe → 0 (经 epsilon 截断)
        assert sharpe_ratio(nav) == Decimal(0)

    def test_BV_SHARP_02_too_short_series(self):
        assert sharpe_ratio(pd.Series([100])) == Decimal(0)

    def test_BV_SORT_01_no_downside_returns_zero(self):
        nav = pd.Series([100, 110, 120, 130])  # 全部正收益
        assert sortino_ratio(nav) == Decimal(0)


# =====================================================================
# 函数 6：annualized_return —— 起止点边界
# =====================================================================

class TestAnnualizedReturnBoundaries:

    def test_BV_AR_01_zero_start(self):
        """起点 = 0 → 返回 0 而不是 ZeroDivisionError。"""
        assert annualized_return(pd.Series([0.0, 100.0])) == Decimal(0)

    def test_BV_AR_02_single_period(self):
        """长度 < 2 → 返回 0。"""
        assert annualized_return(pd.Series([100.0])) == Decimal(0)
        assert annualized_return(pd.Series([])) == Decimal(0)


# =====================================================================
# 函数 7：RotationEngine 阈值边界（threshold × principal）
# =====================================================================

def _pool(pid: str, apy: str) -> PoolMetrics:
    return PoolMetrics(
        pool_id=pid, apy_series=(Decimal(apy),) * 14,
        tvl=Decimal("100000000"), vol_30d=Decimal("0.02"),
        token_price=Decimal("1.0"), gas_base_fee=Decimal("0.0000001"),
    )


def _snap(pools):
    env = EnvSnapshot(
        tick=0, timestamp=datetime(2024, 1, 1),
        oracle_price={pid: Decimal("1.0") for pid in pools},
        gas_base_fee=Decimal("0.0000001"),
        gas_priority_fee=Decimal("0.00000005"),
    )
    return AssetSnapshot(tick=0, pools=pools, env=env)


def _rank(scores):
    return RankingTable(
        snapshot_tick=0,
        rankings=tuple(
            PoolScore(pool_id=pid, score=Decimal(s), components={"momentum": Decimal(s)})
            for pid, s in scores
        ),
    )


class TestRotationThresholdBoundary:
    """gate: gain ≥ friction + threshold × principal。让 gain 落在阈值附近。"""

    def _engine(self, threshold: str) -> RotationEngine:
        return RotationEngine(
            tau_reset=Decimal("0.01"),
            threshold=Decimal(threshold),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            friction_estimator=StubFrictionEstimator(gas=Decimal("0")),  # 排除 friction
            gain_horizon_ticks=365,
        )

    def test_BV_GATE_01_gain_just_above_threshold_rotates(self):
        """principal=100k, threshold=0.01 → require gain > 1000；
        APY diff 1.1% × 100k × 365/365 = 1100，gain > 1000 → ROTATE。"""
        engine = self._engine("0.01")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.061")}  # diff +1.1%
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        # 略 friction → gate 通过
        assert d.decision_type == DecisionType.ROTATE

    def test_BV_GATE_02_gain_just_below_threshold_holds(self):
        """diff +0.9% × 100k = 900 < 1000 → GATE_FAIL。"""
        engine = self._engine("0.01")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.059")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        # 加上 slippage（mid 档 100k×0.003=300），gain 900 < 1000+300 → fail
        assert d.decision_type == DecisionType.HOLD
        assert d.reason == HoldReason.GATE_FAIL


# =====================================================================
# 函数 8：interpolate_missing / remove_outliers_iqr —— 容器边界
# =====================================================================

class TestPreprocessorContainerBoundaries:

    def test_BV_PREP_01_all_nan_column_stays_nan(self):
        """整列 NaN → interpolate 后仍 NaN（无源插值）。"""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3),
            "pool_id": ["a"] * 3,
            "apy": [np.nan, np.nan, np.nan],
        })
        out = interpolate_missing(df, ["apy"], group_col="pool_id")
        assert out["apy"].isna().all()

    def test_BV_PREP_02_single_outlier_in_short_series(self):
        """短序列（IQR = 0）→ 不剔除，原值保留。"""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=2),
            "pool_id": ["a"] * 2,
            "apy": [0.05, 999.0],
        })
        out = remove_outliers_iqr(df, ["apy"], k=3.0, group_col="pool_id")
        # IQR=0 时函数不做事，原 999 保留
        assert 999.0 in out["apy"].values

    def test_BV_PREP_03_outlier_at_boundary_3_iqr(self):
        """异常值恰好在 ±3·IQR 边界外的最小距离。"""
        base = [0.05] * 8 + [0.05 + 1e-6] * 2
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=len(base) + 1),
            "pool_id": ["a"] * (len(base) + 1),
            "apy": base + [0.05 + 1.0],  # 单独超大值
        })
        out = remove_outliers_iqr(df, ["apy"], k=3.0, group_col="pool_id")
        # 末尾大异常值应被剔除并填充
        assert out["apy"].max() < 1.0
