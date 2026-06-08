"""黑盒：决策表法测试（Decision Table Testing）。

PPT 第 8 讲 §9.2：把决策点的所有「条件组合 → 动作」列成决策表，每行作为一个测试用例。
本文件聚焦 RotationEngine.evaluate 的决策矩阵（也是项目里最关键的决策点）。

────────────────────────────────────────────────────────────────────────
                RotationEngine.evaluate 决策表
────────────────────────────────────────────────────────────────────────
 # | C1 has_position | C2 ranking_empty | C3 top_is_same | C4 data_intact | C5 τ_passes | C6 gate_passes | → Decision
───┼─────────────────┼──────────────────┼────────────────┼────────────────┼─────────────┼────────────────┼────────────────────
DT-01 | NO              | -                | -              | Y              | -           | Y              | ROTATE
DT-02 | NO              | -                | -              | Y              | -           | N              | HOLD(GATE_FAIL)
DT-03 | -               | Y                | -              | -              | -           | -              | HOLD(NO_CANDIDATES)
DT-04 | YES             | N                | Y              | -              | -           | -              | HOLD(SAME_POOL)
DT-05 | YES             | N                | N              | Y              | N           | -              | HOLD(TAU_FAIL)
DT-06 | YES             | N                | N              | Y              | Y           | N              | HOLD(GATE_FAIL)
DT-07 | YES             | N                | N              | Y              | Y           | Y              | ROTATE
DT-08 | YES             | N                | N              | N (apy 空)     | -           | -              | HOLD(DATA_ERROR), state=ERROR
────────────────────────────────────────────────────────────────────────

编号规范：DT-RE-<NN>。每个用例严格构造前置条件以隔离单一行为。
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict

import pytest

from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import (
    DecisionType,
    HoldReason,
    PoolScore,
    Position,
    RankingTable,
    RotationState,
)
from strategy.rotation_engine import RotationEngine
from tests._stubs import StubFrictionEstimator

pytestmark = [pytest.mark.blackbox, pytest.mark.decision_table]


# =====================================================================
# 工厂
# =====================================================================

def _pool(pid: str, apy: str = "0.05", empty_series: bool = False) -> PoolMetrics:
    return PoolMetrics(
        pool_id=pid,
        apy_series=() if empty_series else (Decimal(apy),) * 14,
        tvl=Decimal("100000000"),
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


def _rank(scores) -> RankingTable:
    return RankingTable(
        snapshot_tick=0,
        rankings=tuple(
            PoolScore(pool_id=pid, score=Decimal(s), components={"momentum": Decimal(s)})
            for pid, s in scores
        ),
    )


def _engine(threshold: str = "0.001", tau: str = "0.05",
            gas: str = "10", return_negative: bool = False) -> RotationEngine:
    return RotationEngine(
        tau_reset=Decimal(tau),
        threshold=Decimal(threshold),
        gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
        friction_estimator=StubFrictionEstimator(
            gas=Decimal(gas), return_negative=return_negative,
        ),
        gain_horizon_ticks=30,
    )


# =====================================================================
# 决策表行：每行 1 个用例
# =====================================================================

class TestRotationEngineDecisionTable:

    def test_DT_RE_01_no_position_gate_passes_rotate(self):
        """C1=NO position, gate 通过 → ROTATE 到 top。"""
        engine = _engine(threshold="0.0001")
        pools = {"a": _pool("a", "0.20"), "b": _pool("b", "0.05")}
        pos = Position.empty(initial_cash=Decimal("100000"))
        d = engine.evaluate(pos, _rank([("a", "1.0"), ("b", "0.5")]), _snap(pools))
        assert d.decision_type == DecisionType.ROTATE
        assert d.target_pool_id == "a"

    def test_DT_RE_02_no_position_gate_fails_hold(self):
        """C1=NO position, gate 不通过（gas 极高）→ HOLD(GATE_FAIL)。"""
        engine = _engine(threshold="0.0001", gas="50000")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.05")}
        pos = Position.empty(initial_cash=Decimal("100000"))
        d = engine.evaluate(pos, _rank([("a", "1.0"), ("b", "0.5")]), _snap(pools))
        assert d.decision_type == DecisionType.HOLD
        assert d.reason == HoldReason.GATE_FAIL

    def test_DT_RE_03_empty_ranking_hold_no_candidates(self):
        """C2=YES (ranking 空) → HOLD(NO_CANDIDATES)。"""
        engine = _engine()
        pools = {"a": _pool("a")}
        d = engine.evaluate(
            Position.empty(initial_cash=Decimal("100000")),
            _rank([]),    # 空排名
            _snap(pools),
        )
        assert d.decision_type == DecisionType.HOLD
        assert d.reason == HoldReason.NO_CANDIDATES

    def test_DT_RE_04_top_is_same_as_holding_hold_same_pool(self):
        """C3=YES (top == current pool) → HOLD(SAME_POOL)。"""
        engine = _engine()
        pools = {"a": _pool("a", "0.10"), "b": _pool("b", "0.05")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("a", "1.0"), ("b", "0.5")]), _snap(pools))
        assert d.decision_type == DecisionType.HOLD
        assert d.reason == HoldReason.SAME_POOL

    def test_DT_RE_05_tau_fails_hold_tau_fail(self):
        """C5=NO (τ 不通过：score 差 + APY 差都小) → HOLD(TAU_FAIL)。"""
        engine = _engine(tau="0.10")  # 要求 +10%
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.052")}  # APY 仅差 4%
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        # score 差也小：0.51 - 0.50 = 0.01 < τ=0.10
        d = engine.evaluate(pos, _rank([("b", "0.51"), ("a", "0.50")]), _snap(pools))
        assert d.decision_type == DecisionType.HOLD
        assert d.reason == HoldReason.TAU_FAIL

    def test_DT_RE_06_tau_passes_gate_fails_hold(self):
        """C5=YES, C6=NO → HOLD(GATE_FAIL)。"""
        engine = _engine(threshold="0.001", gas="50000")  # gas 高 → gate 一定不通过
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.50")}  # APY 大差
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.decision_type == DecisionType.HOLD
        assert d.reason == HoldReason.GATE_FAIL

    def test_DT_RE_07_tau_passes_gate_passes_rotate(self):
        """C5=YES, C6=YES → ROTATE。"""
        engine = _engine(threshold="0.0001", gas="10")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.50")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.decision_type == DecisionType.ROTATE
        assert d.target_pool_id == "b"

    def test_DT_RE_08_data_integrity_error_hold_with_error_state(self):
        """C4=NO (apy_series 空) → HOLD(DATA_ERROR) + state = ERROR。"""
        engine = _engine()
        pools = {
            "a": _pool("a", "0.05", empty_series=True),   # 当前持仓池缺数据
            "b": _pool("b", "0.20"),
        }
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.decision_type == DecisionType.HOLD
        assert d.reason == HoldReason.DATA_ERROR
        assert engine.state == RotationState.ERROR


# =====================================================================
# ReinvestEngine 决策表
# ────────────────────────────────────────────────────────────────────
# # | C1 has_position | C2 pending>0 | C3 gain > gas×premium | → Decision
# ──┼─────────────────┼──────────────┼───────────────────────┼─────────────
# DT-RI-01 | NO              | -            | -                     | False, "NO_POSITION"
# DT-RI-02 | YES             | NO           | -                     | False, "NO_REWARDS"
# DT-RI-03 | YES             | YES          | NO                    | False, "NEGATIVE_NET"
# DT-RI-04 | YES             | YES          | YES                   | True, "OK"
# =====================================================================

from strategy.reinvest_engine import ReinvestEngine    # noqa: E402


class TestReinvestEngineDecisionTable:

    @pytest.fixture
    def fast_engine(self):
        """gas 低 + window 长 → 容易触发复投。"""
        return ReinvestEngine(
            friction_estimator=StubFrictionEstimator(gas=Decimal("1")),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            reinvest_window=180,
            risk_premium_multiplier=Decimal("1.5"),
        )

    @pytest.fixture
    def slow_engine(self):
        """gas 高 → 几乎不触发。"""
        return ReinvestEngine(
            friction_estimator=StubFrictionEstimator(gas=Decimal("1000")),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            reinvest_window=30,
            risk_premium_multiplier=Decimal("1.5"),
        )

    def test_DT_RI_01_no_position(self, fast_engine):
        snap = _snap({"a": _pool("a", "0.20")})
        d = fast_engine.evaluate(Position.empty(initial_cash=Decimal("100000")), snap)
        assert d.do_reinvest is False
        assert d.reason == "NO_POSITION"

    def test_DT_RI_02_no_pending_reward(self, fast_engine):
        snap = _snap({"a": _pool("a", "0.20")})
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = fast_engine.evaluate(pos, snap)
        assert d.do_reinvest is False
        assert d.reason == "NO_REWARDS"

    def test_DT_RI_03_negative_net(self, slow_engine):
        snap = _snap({"a": _pool("a", "0.05")})
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal("10"), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = slow_engine.evaluate(pos, snap)
        assert d.do_reinvest is False
        assert d.reason == "NEGATIVE_NET"

    def test_DT_RI_04_positive_net_triggers(self, fast_engine):
        snap = _snap({"a": _pool("a", "0.20")})
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal("1000"), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = fast_engine.evaluate(pos, snap)
        assert d.do_reinvest is True
        assert d.reason == "OK"
