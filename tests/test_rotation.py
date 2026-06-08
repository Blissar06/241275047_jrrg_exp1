"""Phase 2 第二段：RotationEngine 单元测试。

覆盖：
  - τ-reset HOLD（TAU_FAIL）
  - gate HOLD（GATE_FAIL）
  - friction 异常 / 负值的 fallback（E-RT-004）
  - DataIntegrityError → state=ERROR + HOLD（E-RT-001）
  - commit() 正确更新 Position 与 TradeLog
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict, Tuple

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
    TradeLog,
)
from strategy.rotation_engine import RotationEngine
from tests._stubs import StubFrictionEstimator


# ----- helpers -----

def _pool(pid: str, apy: str, tvl: str = "1000000") -> PoolMetrics:
    return PoolMetrics(
        pool_id=pid,
        apy_series=(Decimal(apy),) * 14,
        tvl=Decimal(tvl),
        vol_30d=Decimal("0.02"),
        token_price=Decimal("1.0"),
        gas_base_fee=Decimal("20"),
    )


def _snap(pools: Dict[str, PoolMetrics], tick: int = 0) -> AssetSnapshot:
    env = EnvSnapshot(
        tick=tick,
        timestamp=datetime(2024, 1, 1),
        oracle_price={pid: Decimal("1.0") for pid in pools},
        gas_base_fee=Decimal("20"),
        gas_priority_fee=Decimal("1.5"),
    )
    return AssetSnapshot(tick=tick, pools=pools, env=env)


def _ranking(scores: list[Tuple[str, str]], tick: int = 0) -> RankingTable:
    rankings = tuple(
        PoolScore(pool_id=pid, score=Decimal(s), components={"momentum": Decimal(s)})
        for pid, s in scores
    )
    return RankingTable(snapshot_tick=tick, rankings=rankings)


def _make_engine(
    tau: str = "0.05",
    threshold: str = "0.01",
    gas: str = "10",
    horizon: int = 30,
) -> Tuple[RotationEngine, StubFrictionEstimator]:
    friction = StubFrictionEstimator(gas=Decimal(gas))
    engine = RotationEngine(
        tau_reset=Decimal(tau),
        threshold=Decimal(threshold),
        gain_estimator=APYDeltaGainEstimator(),
        friction_estimator=friction,
        gain_horizon_ticks=horizon,
    )
    return engine, friction


# ----- tau-reset -----

def test_tau_fail_when_target_apy_close_to_current():
    """target 与 current 在 APY 与 score 两个维度都接近 → TAU_FAIL。"""
    engine, _ = _make_engine(tau="0.10")  # 需 +10% 才换
    pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.052")}
    snap = _snap(pools)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal(0), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)
    # score gap = 0.01 < τ=0.10；APY gap = 0.04 < τ → 双双失败
    rk = _ranking([("b", "0.51"), ("a", "0.50")])

    decision = engine.evaluate(pos, rk, snap)

    assert decision.decision_type == DecisionType.HOLD
    assert decision.reason == HoldReason.TAU_FAIL
    assert engine.state == RotationState.HOLDING


def test_tau_pass_when_no_current_position():
    """初始无持仓时必须开仓（APY 差够大以越过门槛）。"""
    engine, _ = _make_engine(threshold="0.001")
    pools = {"a": _pool("a", "0.20"), "b": _pool("b", "0.10")}
    snap = _snap(pools)
    pos = Position.empty(initial_cash=Decimal("100000"))
    rk = _ranking([("a", "1.0"), ("b", "0.5")])

    decision = engine.evaluate(pos, rk, snap)

    assert decision.decision_type == DecisionType.ROTATE
    assert decision.target_pool_id == "a"


def test_same_pool_returns_hold_same_pool():
    engine, _ = _make_engine()
    pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.04")}
    snap = _snap(pools)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal(0), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)
    rk = _ranking([("a", "1.0"), ("b", "0.5")])

    d = engine.evaluate(pos, rk, snap)
    assert d.decision_type == DecisionType.HOLD
    assert d.reason == HoldReason.SAME_POOL


# ----- gate -----

def test_gate_fail_when_friction_eats_gain():
    """target APY 高但摩擦 + threshold 吃光增益 → GATE_FAIL。"""
    engine, _ = _make_engine(gas="2000", threshold="0.01")
    # APY 差 0.5% 太小，30 天预期增益约 100000 * 0.005 * 30/365 ≈ 41
    pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.055")}
    snap = _snap(pools)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal(0), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)
    rk = _ranking([("b", "1.0"), ("a", "0.5")])

    d = engine.evaluate(pos, rk, snap)
    assert d.decision_type == DecisionType.HOLD
    assert d.reason == HoldReason.GATE_FAIL
    # 校验关键字段都被填好（便于报表归因）
    assert d.expected_gain > 0
    assert d.estimated_friction.total > 0
    assert d.threshold_required > 0


def test_gate_pass_when_apy_jump_large_enough():
    engine, _ = _make_engine(gas="10", threshold="0.001")
    pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.20")}  # +15% 跳变
    snap = _snap(pools)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal(0), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)
    rk = _ranking([("b", "1.0"), ("a", "0.5")])

    d = engine.evaluate(pos, rk, snap)
    assert d.decision_type == DecisionType.ROTATE
    assert d.target_pool_id == "b"
    assert engine.state == RotationState.RANKED
    assert engine.pending_decision is d


# ----- friction error fallback (E-RT-004) -----

def test_friction_negative_falls_back_to_cache():
    """先来一次正常调用建立缓存，再让 stub 返回负值，应该回退到缓存值。"""
    engine, friction = _make_engine(gas="10", threshold="0.001")
    pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.20")}
    snap = _snap(pools, tick=0)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal(0), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)
    rk = _ranking([("b", "1.0"), ("a", "0.5")])

    d1 = engine.evaluate(pos, rk, snap)
    cached_total = d1.estimated_friction.total

    # 切到 tick 1，friction 估算返回负值
    friction.return_negative = True
    snap2 = _snap(pools, tick=1)
    d2 = engine.evaluate(pos, rk, snap2)
    # 仍能给出合理结果（用了缓存）
    assert d2.estimated_friction.total == cached_total


def test_friction_raise_falls_back_to_zero_when_no_cache():
    """首次调用就抛异常，无缓存可用 → 退回到 0 friction。"""
    friction = StubFrictionEstimator(raise_on_call=True)
    engine = RotationEngine(
        tau_reset=Decimal("0.05"),
        threshold=Decimal("0.001"),
        gain_estimator=APYDeltaGainEstimator(),
        friction_estimator=friction,
    )
    pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.20")}
    snap = _snap(pools)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal(0), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)
    rk = _ranking([("b", "1.0"), ("a", "0.5")])

    d = engine.evaluate(pos, rk, snap)
    # friction=0 → gate 必然通过（threshold × principal 仍要满足，但 apy 差大）
    assert d.estimated_friction.total == Decimal(0)


# ----- DataIntegrityError (E-RT-001) -----

def test_data_integrity_error_fallbacks_to_hold():
    engine, _ = _make_engine()
    pools = {
        "a": PoolMetrics(  # apy_series 为空
            pool_id="a", apy_series=(),
            tvl=Decimal("1000000"), vol_30d=Decimal(0),
            token_price=Decimal(1), gas_base_fee=Decimal(0),
        ),
        "b": _pool("b", "0.20"),
    }
    snap = _snap(pools)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal(0), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)
    rk = _ranking([("b", "1.0"), ("a", "0.5")])

    d = engine.evaluate(pos, rk, snap)
    assert d.decision_type == DecisionType.HOLD
    assert d.reason == HoldReason.DATA_ERROR
    assert engine.state == RotationState.ERROR


# ----- commit -----

def test_commit_updates_position_and_emits_trade_log():
    engine, _ = _make_engine(gas="50", threshold="0.001")
    pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.20")}
    snap = _snap(pools, tick=5)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal("100"), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)
    rk = _ranking([("b", "1.0"), ("a", "0.5")])

    decision = engine.evaluate(pos, rk, snap)
    assert decision.decision_type == DecisionType.ROTATE

    new_pos, log = engine.commit(decision, pos, snap)

    assert isinstance(log, TradeLog)
    assert new_pos.pool_id == "b"
    assert new_pos.opened_tick == 5
    assert new_pos.pending_reward == Decimal(0)
    # principal = (100000 + 100) - friction.total
    expected_principal = Decimal("100100") - decision.estimated_friction.total
    assert new_pos.principal == expected_principal
    # 状态机回到 IDLE
    assert engine.state == RotationState.IDLE
    assert engine.pending_decision is None


def test_commit_rejects_hold_decision():
    engine, _ = _make_engine(gas="2000")  # 强制 GATE_FAIL
    pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.052")}
    snap = _snap(pools)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal(0), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)
    rk = _ranking([("b", "1.0"), ("a", "0.5")])

    decision = engine.evaluate(pos, rk, snap)
    with pytest.raises(ValueError, match="ROTATE"):
        engine.commit(decision, pos, snap)


# ----- 复现性 -----

def test_evaluate_is_deterministic():
    """相同输入两次 evaluate() 输出完全一致（NFR-02）。"""
    engine_a, _ = _make_engine()
    engine_b, _ = _make_engine()
    pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.20")}
    snap = _snap(pools)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal(0), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)
    rk = _ranking([("b", "1.0"), ("a", "0.5")])

    d1 = engine_a.evaluate(pos, rk, snap)
    d2 = engine_b.evaluate(pos, rk, snap)
    assert d1.decision_type == d2.decision_type
    assert d1.target_pool_id == d2.target_pool_id
    assert d1.expected_gain == d2.expected_gain
    assert d1.estimated_friction == d2.estimated_friction
