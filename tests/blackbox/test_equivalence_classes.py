"""黑盒：等价类划分测试（Equivalence Class Partitioning）。

PPT 第 8 讲 §9.2：把输入空间划分为有效等价类（valid）+ 无效等价类（invalid），
每类至少 1 个用例。编号规范：EC-<函数缩写>-<序号>，例如 EC-CAP-V1。

被测函数清单：
  1. apply_capacity_decay (CAP)
  2. estimate_slippage   (SLIP)
  3. ewma                (EWM)
  4. RotationEngine._gate (内部门槛函数，通过 evaluate 黑盒触发)  (GATE)
  5. ReinvestEngine.evaluate                                    (REI)
  6. WeightConfig.normalized                                    (WCFG)
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict

import pytest

from backtest.cost_model import FrictionEstimator
from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from data_model.preprocessor import apply_capacity_decay
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import (
    DecisionType,
    HoldReason,
    PoolScore,
    Position,
    RankingTable,
    WeightConfig,
)
from strategy.reinvest_engine import ReinvestEngine
from strategy.rotation_engine import RotationEngine
from strategy.scorers.momentum import ewma
from tests._stubs import StubFrictionEstimator

pytestmark = [pytest.mark.blackbox, pytest.mark.equivalence]


# =====================================================================
# 函数 1：apply_capacity_decay (CAP)
# =====================================================================
# 输入空间：
#   apy_nominal  ∈ ℝ ：值类（V1 正常 / I1 负值 —— 此处函数允许任意输入，
#                       负 APY 在借贷池场景合法）
#   tvl          ∈ ℝ⁺：值类（V2 正常 / I2 0 / I3 负 —— 实际不应为负但函数应不崩）
#   capital      ∈ ℝ⁺：值类（V3 = 0 不稀释 / V4 远小于 TVL / V5 ≈ TVL 半稀释
#                            / V6 远大于 TVL 重度稀释 / I4 负数）
#   pool_kind ∈ {"yield","lending"} ：(V7 默认 / V8 lending+utilization)
# =====================================================================

class TestApplyCapacityDecay:

    def test_EC_CAP_V1_zero_capital_no_dilution(self):
        """V1：注入资本 0 → 不稀释，actual = nominal。"""
        out = apply_capacity_decay(Decimal("0.10"), Decimal("1000000"), Decimal(0))
        assert out == Decimal("0.10")

    def test_EC_CAP_V2_small_capital_minor_dilution(self):
        """V2：注入远小于 TVL → 稀释 < 1%。"""
        out = apply_capacity_decay(Decimal("0.10"), Decimal("1000000"), Decimal("1000"))
        assert Decimal("0.099") < out < Decimal("0.10")

    def test_EC_CAP_V3_equal_capital_half_dilution(self):
        """V3：注入 ≈ TVL → 稀释一半。"""
        out = apply_capacity_decay(Decimal("0.10"), Decimal("1000000"), Decimal("1000000"))
        assert out == Decimal("0.05")

    def test_EC_CAP_V4_huge_capital_heavy_dilution(self):
        """V4：注入远大于 TVL → 极度稀释 → APY 趋近 0。"""
        out = apply_capacity_decay(Decimal("0.10"), Decimal("1000000"), Decimal("1000000000"))
        assert out < Decimal("0.001")

    def test_EC_CAP_V5_lending_kind_under_kink(self):
        """V5：借贷池利用率 < kink (0.8) → 在通用模型上再轻微调整。"""
        out = apply_capacity_decay(
            Decimal("0.10"), Decimal("1000000"), Decimal("100000"),
            pool_kind="lending", utilization=Decimal("0.5"),
        )
        assert out > Decimal(0)

    def test_EC_CAP_V6_lending_kind_above_kink_penalized(self):
        """V6：借贷池利用率 ≥ kink → 边际惩罚显著高于 < kink 场景。"""
        below = apply_capacity_decay(
            Decimal("0.10"), Decimal("1000000"), Decimal("100000"),
            pool_kind="lending", utilization=Decimal("0.5"),
        )
        above = apply_capacity_decay(
            Decimal("0.10"), Decimal("1000000"), Decimal("100000"),
            pool_kind="lending", utilization=Decimal("0.95"),
        )
        assert above < below

    def test_EC_CAP_I1_zero_tvl_returns_zero(self):
        """I1：TVL = 0 → 函数应返回 0 而不是除零异常。"""
        out = apply_capacity_decay(Decimal("0.10"), Decimal(0), Decimal(0))
        assert out == Decimal(0)


# =====================================================================
# 函数 2：FrictionEstimator.estimate_slippage (SLIP)
# =====================================================================
# 输入空间 (trade_size, tvl)：
#   ratio = trade / tvl
#   V1: ratio < threshold_low (0.01)     → low_rate
#   V2: low ≤ ratio < high_threshold     → mid_rate
#   V3: ratio ≥ high_threshold (0.05)    → high_rate
#   I1: tvl ≤ 0                          → 直接走 high_rate
#   I2: trade_size ≤ 0                   → 0 滑点
# =====================================================================

class TestEstimateSlippage:

    @pytest.fixture
    def fe(self):
        return FrictionEstimator()

    def test_EC_SLIP_V1_below_low_threshold_uses_low_rate(self, fe):
        cost = fe.estimate_slippage(Decimal("5000"), Decimal("1000000"))  # 0.5%
        assert cost == Decimal("5000") * Decimal("0.001")

    def test_EC_SLIP_V2_between_thresholds_uses_mid_rate(self, fe):
        cost = fe.estimate_slippage(Decimal("30000"), Decimal("1000000"))  # 3%
        assert cost == Decimal("30000") * Decimal("0.003")

    def test_EC_SLIP_V3_above_high_threshold_uses_high_rate(self, fe):
        cost = fe.estimate_slippage(Decimal("100000"), Decimal("1000000"))  # 10%
        assert cost == Decimal("100000") * Decimal("0.008")

    def test_EC_SLIP_I1_zero_tvl_uses_high_rate(self, fe):
        cost = fe.estimate_slippage(Decimal("100"), Decimal(0))
        assert cost == Decimal("100") * Decimal("0.008")

    def test_EC_SLIP_I2_zero_trade_size_returns_zero(self, fe):
        assert fe.estimate_slippage(Decimal(0), Decimal("1000000")) == Decimal(0)


# =====================================================================
# 函数 3：ewma (EWM)
# =====================================================================
# 输入空间：
#   series：(V1 单元素 / V2 多元素恒定 / V3 多元素递增 / I1 空)
#   lam：  (V4 ∈ (0,1) 正常 / I2 ≤ 0 / I3 ≥ 1)
# =====================================================================

class TestEwma:

    def test_EC_EWM_V1_single_element_returns_self(self):
        assert ewma((Decimal("0.05"),), Decimal("0.85")) == Decimal("0.05")

    def test_EC_EWM_V2_constant_series_returns_constant(self):
        assert ewma((Decimal("0.05"),) * 5, Decimal("0.85")) == Decimal("0.05")

    def test_EC_EWM_V3_monotone_series_returns_between_first_and_last(self):
        series = tuple(Decimal(str(0.01 * i)) for i in range(1, 11))
        out = ewma(series, Decimal("0.85"))
        assert series[0] < out < series[-1]

    def test_EC_EWM_I1_empty_returns_zero(self):
        assert ewma((), Decimal("0.85")) == Decimal(0)

    @pytest.mark.parametrize("invalid_lam", [Decimal(0), Decimal(1), Decimal("1.5"), Decimal("-0.1")])
    def test_EC_EWM_I2_invalid_lambda_raises(self, invalid_lam):
        with pytest.raises(ValueError):
            ewma((Decimal("0.05"), Decimal("0.06")), invalid_lam)


# =====================================================================
# 函数 4：RotationEngine.evaluate → _gate（黑盒通过 evaluate 验证）(GATE)
# =====================================================================
# 输入空间（expected_gain, friction.total, threshold × principal）：
#   V1: gain >> friction + thr × P     → ROTATE
#   V2: gain ≈ friction + thr × P 但低 → HOLD(GATE_FAIL)
#   V3: gain < friction                → HOLD(GATE_FAIL)
# =====================================================================

def _pool(pid: str, apy: str) -> PoolMetrics:
    return PoolMetrics(
        pool_id=pid,
        apy_series=(Decimal(apy),) * 14,
        tvl=Decimal("100000000"),    # 大 TVL → 滑点档低
        vol_30d=Decimal("0.02"),
        token_price=Decimal("1.0"),
        gas_base_fee=Decimal("0.0000001"),
    )


def _snap(pools: Dict[str, PoolMetrics], tick: int = 0) -> AssetSnapshot:
    env = EnvSnapshot(
        tick=tick, timestamp=datetime(2024, 1, 1),
        oracle_price={pid: Decimal("1.0") for pid in pools},
        gas_base_fee=Decimal("0.0000001"),
        gas_priority_fee=Decimal("0.00000005"),
    )
    return AssetSnapshot(tick=tick, pools=pools, env=env)


def _ranking(scores):
    rankings = tuple(
        PoolScore(pool_id=pid, score=Decimal(s), components={"momentum": Decimal(s)})
        for pid, s in scores
    )
    return RankingTable(snapshot_tick=0, rankings=rankings)


class TestGate:

    def _engine(self, threshold: str, gas: str = "10") -> RotationEngine:
        return RotationEngine(
            tau_reset=Decimal("0.01"),
            threshold=Decimal(threshold),
            gain_estimator=APYDeltaGainEstimator(),
            friction_estimator=StubFrictionEstimator(gas=Decimal(gas)),
            gain_horizon_ticks=30,
        )

    def test_EC_GATE_V1_gain_far_above_friction_rotates(self):
        engine = self._engine(threshold="0.0001")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.50")}
        snap = _snap(pools)
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _ranking([("b", "1.0"), ("a", "0.5")]), snap)
        assert d.decision_type == DecisionType.ROTATE

    def test_EC_GATE_V2_gain_marginally_below_threshold_holds(self):
        # threshold = 1% × principal = 1000；让 gain ≈ friction + 800
        engine = self._engine(threshold="0.01", gas="100")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.055")}  # APY 差 0.5%
        snap = _snap(pools)
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _ranking([("b", "1.0"), ("a", "0.5")]), snap)
        assert d.decision_type == DecisionType.HOLD
        assert d.reason == HoldReason.GATE_FAIL

    def test_EC_GATE_V3_gain_negative_due_to_high_friction(self):
        engine = self._engine(threshold="0.001", gas="50000")  # 极高 gas
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.052")}
        snap = _snap(pools)
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _ranking([("b", "1.0"), ("a", "0.5")]), snap)
        assert d.decision_type == DecisionType.HOLD
        assert d.reason == HoldReason.GATE_FAIL


# =====================================================================
# 函数 5：ReinvestEngine.evaluate (REI)
# =====================================================================
# 等价类（按 do_reinvest 与 reason）：
#   V1: 有持仓 + pending > 0 + gain > gas×premium  → do_reinvest=True, "OK"
#   V2: 有持仓 + pending = 0                       → False, "NO_REWARDS"
#   V3: 有持仓 + pending > 0 + gain ≤ threshold    → False, "NEGATIVE_NET"
#   I1: 无持仓                                      → False, "NO_POSITION"
# =====================================================================

class TestReinvestEvaluate:

    @pytest.fixture
    def engine(self):
        return ReinvestEngine(
            friction_estimator=StubFrictionEstimator(gas=Decimal("1")),
            gain_estimator=APYDeltaGainEstimator(),
            reinvest_window=180,
            risk_premium_multiplier=Decimal("1.5"),
        )

    def test_EC_REI_V1_positive_net_reinvests(self, engine):
        snap = _snap({"a": _pool("a", "0.20")})
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal("1000"), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, snap)
        assert d.do_reinvest is True
        assert d.reason == "OK"

    def test_EC_REI_V2_no_rewards_skips(self, engine):
        snap = _snap({"a": _pool("a", "0.20")})
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, snap)
        assert d.do_reinvest is False
        assert d.reason == "NO_REWARDS"

    def test_EC_REI_V3_negative_net_skips(self):
        engine = ReinvestEngine(
            friction_estimator=StubFrictionEstimator(gas=Decimal("1000")),  # 高 gas
            gain_estimator=APYDeltaGainEstimator(),
            reinvest_window=30, risk_premium_multiplier=Decimal("1.5"),
        )
        snap = _snap({"a": _pool("a", "0.05")})
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal("10"), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, snap)
        assert d.do_reinvest is False
        assert d.reason == "NEGATIVE_NET"

    def test_EC_REI_I1_no_position_returns_no_position(self, engine):
        snap = _snap({"a": _pool("a", "0.20")})
        pos = Position.empty(initial_cash=Decimal("100000"))
        d = engine.evaluate(pos, snap)
        assert d.do_reinvest is False
        assert d.reason == "NO_POSITION"


# =====================================================================
# 函数 6：WeightConfig.normalized (WCFG)
# =====================================================================
# 等价类（权重 dict 的总和）：
#   V1: 总和 > 0                          → 归一化为总和 1
#   V2: 总和 = 1（已归一化）              → 幂等返回近似相同
#   I1: 总和 = 0                          → 抛 ValueError
#   I2: 总和 < 0（含负权重）              → 抛 ValueError
# =====================================================================

class TestWeightConfigNormalized:

    def test_EC_WCFG_V1_positive_sum_normalizes_to_one(self):
        cfg = WeightConfig({"a": Decimal("2"), "b": Decimal("3"), "c": Decimal("5")})
        n = cfg.normalized()
        assert sum(n.weights.values()) == Decimal(1)

    def test_EC_WCFG_V2_already_normalized_is_idempotent(self):
        cfg = WeightConfig({"a": Decimal("0.5"), "b": Decimal("0.5")})
        n = cfg.normalized()
        assert n.weights["a"] == Decimal("0.5")

    def test_EC_WCFG_I1_zero_sum_raises(self):
        with pytest.raises(ValueError):
            WeightConfig({"a": Decimal(0)}).normalized()

    def test_EC_WCFG_I2_negative_sum_raises(self):
        with pytest.raises(ValueError):
            WeightConfig({"a": Decimal("-1"), "b": Decimal("0.5")}).normalized()
