"""Phase 2 第二段：ReinvestEngine 单元测试。"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict

import pytest

from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import Position, ReinvestLog
from strategy.reinvest_engine import ReinvestEngine
from tests._stubs import StubFrictionEstimator


def _pool(pid: str, apy: str) -> PoolMetrics:
    return PoolMetrics(
        pool_id=pid,
        apy_series=(Decimal(apy),) * 14,
        tvl=Decimal("1000000"),
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


def _make_engine(
    gas: str = "5",
    window: int = 30,
    risk_premium: str = "1.5",
) -> ReinvestEngine:
    return ReinvestEngine(
        friction_estimator=StubFrictionEstimator(gas=Decimal(gas)),
        gain_estimator=APYDeltaGainEstimator(),
        reinvest_window=window,
        risk_premium_multiplier=Decimal(risk_premium),
    )


def test_no_position_returns_no_reinvest():
    engine = _make_engine()
    snap = _snap({"a": _pool("a", "0.05")})
    pos = Position.empty(initial_cash=Decimal("100000"))

    d = engine.evaluate(pos, snap)
    assert d.do_reinvest is False
    assert d.reason == "NO_POSITION"


def test_no_pending_reward_returns_no_reinvest():
    engine = _make_engine()
    snap = _snap({"a": _pool("a", "0.05")})
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal(0), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)

    d = engine.evaluate(pos, snap)
    assert d.do_reinvest is False
    assert d.reason == "NO_REWARDS"


def test_negative_net_when_gas_too_high():
    """gas × premium 超过预期增益 → 不复投。"""
    engine = _make_engine(gas="100")    # 高 gas
    snap = _snap({"a": _pool("a", "0.05")})
    # pending=10, apy=5%, window=30 → expected≈10*0.05*30/365≈0.04
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal("10"), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)

    d = engine.evaluate(pos, snap)
    assert d.do_reinvest is False
    assert d.reason == "NEGATIVE_NET"


def test_positive_net_triggers_reinvest():
    """gas 低、pending 大 → 触发复投。"""
    engine = _make_engine(gas="1", window=180)
    snap = _snap({"a": _pool("a", "0.20")})  # 20% APY
    # pending=1000, apy=20%, window=180 → expected=1000*0.2*180/365≈98.6 > 1*1.5
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal("1000"), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)

    d = engine.evaluate(pos, snap)
    assert d.do_reinvest is True
    assert d.reason == "OK"
    assert d.expected_gain > d.gas_cost


def test_commit_reinvest_compounds_reward_into_principal():
    engine = _make_engine(gas="1", window=180)
    snap = _snap({"a": _pool("a", "0.20")}, tick=42)
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal("1000"), cash=Decimal("100"),
                   opened_tick=0, last_compound_tick=0)

    d = engine.evaluate(pos, snap)
    new_pos, log = engine.commit_reinvest(d, pos, snap)

    assert isinstance(log, ReinvestLog)
    assert log.tick == 42
    assert log.pool_id == "a"
    # gas 从 cash 优先扣（cash=100 足够支付 gas=1）
    assert new_pos.cash == Decimal("99")
    # 全部 pending 注入 principal
    assert new_pos.principal == Decimal("101000")
    assert new_pos.pending_reward == Decimal(0)
    assert new_pos.last_compound_tick == 42


def test_commit_reinvest_pays_gas_from_pending_when_cash_insufficient():
    engine = _make_engine(gas="50")
    snap = _snap({"a": _pool("a", "0.20")}, tick=10)
    # cash=10 不够支付 gas=50，需从 pending=1000 中再扣 40
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal("1000"), cash=Decimal("10"),
                   opened_tick=0, last_compound_tick=0)

    # 强制做大 window 以确保通过门槛
    engine = _make_engine(gas="50", window=365)
    d = engine.evaluate(pos, snap)
    assert d.do_reinvest is True

    new_pos, _ = engine.commit_reinvest(d, pos, snap)
    assert new_pos.cash == Decimal(0)
    # principal += pending - (gas - cash) = 1000 - 40 = 960
    assert new_pos.principal == Decimal("100960")


def test_commit_rejects_no_reinvest_decision():
    engine = _make_engine(gas="100")
    snap = _snap({"a": _pool("a", "0.05")})
    pos = Position(pool_id="a", principal=Decimal("100000"),
                   pending_reward=Decimal("10"), cash=Decimal(0),
                   opened_tick=0, last_compound_tick=0)

    d = engine.evaluate(pos, snap)
    assert d.do_reinvest is False
    with pytest.raises(ValueError):
        engine.commit_reinvest(d, pos, snap)
