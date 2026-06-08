"""Phase 3 命令 3-2：EventInjector 单元测试。"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict

import pytest
import yaml

from backtest.event_injector import EventInjector, EventType, StressEvent
from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics


def _pool(pid: str, apy: str = "0.05", tvl: str = "1000000") -> PoolMetrics:
    return PoolMetrics(
        pool_id=pid,
        apy_series=(Decimal(apy),) * 5,
        tvl=Decimal(tvl),
        vol_30d=Decimal("0.02"),
        token_price=Decimal("1.0"),
        gas_base_fee=Decimal("20"),
    )


def _snap(pools: Dict[str, PoolMetrics], tick: int = 0, base_fee: str = "20") -> AssetSnapshot:
    env = EnvSnapshot(
        tick=tick,
        timestamp=datetime(2024, 1, 1),
        oracle_price={pid: Decimal("1.0") for pid in pools},
        gas_base_fee=Decimal(base_fee),
        gas_priority_fee=Decimal("1.5"),
    )
    return AssetSnapshot(tick=tick, pools=pools, env=env)


# ----- StressEvent dataclass -----

def test_stress_event_validates_duration():
    with pytest.raises(ValueError):
        StressEvent(
            event_type=EventType.GAS_SPIKE,
            start_tick=0, duration=0, impact_ratio=Decimal("1.0"),
        )


def test_stress_event_pool_exploit_requires_target():
    with pytest.raises(ValueError):
        StressEvent(
            event_type=EventType.POOL_EXPLOIT,
            start_tick=0, duration=1, impact_ratio=Decimal("0.5"),
            target_pool_id=None,
        )


def test_stress_event_is_active_window():
    ev = StressEvent(EventType.GAS_SPIKE, start_tick=10, duration=3, impact_ratio=Decimal(1))
    assert not ev.is_active(9)
    assert ev.is_active(10)
    assert ev.is_active(12)
    assert not ev.is_active(13)


# ----- apply -----

def test_apply_no_active_returns_same_snapshot():
    inj = EventInjector([
        StressEvent(EventType.GAS_SPIKE, 100, 5, Decimal(1)),
    ])
    snap = _snap({"a": _pool("a")}, tick=50)
    out = inj.apply(snap)
    assert out is snap   # 性能优化：无事件时直接复用


def test_gas_spike_doubles_gas_when_impact_1():
    inj = EventInjector([
        StressEvent(EventType.GAS_SPIKE, 0, 1, Decimal("1.0")),
    ])
    snap = _snap({"a": _pool("a")}, tick=0, base_fee="20")
    out = inj.apply(snap)
    assert out.env.gas_base_fee == Decimal("40")
    # priority_fee 也按比例提升
    assert out.env.gas_priority_fee == Decimal("3.0")


def test_pool_exploit_drops_apy_and_tvl():
    inj = EventInjector([
        StressEvent(EventType.POOL_EXPLOIT, 0, 1, Decimal("0.9"), target_pool_id="b"),
    ])
    snap = _snap({"a": _pool("a"), "b": _pool("b", apy="0.10", tvl="1000000")})
    out = inj.apply(snap)
    # b: APY × 0.1（最后一项），tvl × 0.1
    assert out.pools["b"].apy_series[-1] == Decimal("0.10") * Decimal("0.1")
    assert out.pools["b"].tvl == Decimal("100000")
    # a 不受影响
    assert out.pools["a"].tvl == snap.pools["a"].tvl


def test_pool_exploit_preserves_history():
    inj = EventInjector([
        StressEvent(EventType.POOL_EXPLOIT, 0, 1, Decimal("0.9"), target_pool_id="a"),
    ])
    snap = _snap({"a": _pool("a", apy="0.10")}, tick=0)
    out = inj.apply(snap)
    # 前 4 项不变，仅末项被打折
    assert out.pools["a"].apy_series[:-1] == snap.pools["a"].apy_series[:-1]


def test_liquidity_dryup_only_shrinks_tvl():
    inj = EventInjector([
        StressEvent(EventType.LIQUIDITY_DRYUP, 0, 1, Decimal("0.95"), target_pool_id="a"),
    ])
    snap = _snap({"a": _pool("a", apy="0.10", tvl="1000000")})
    out = inj.apply(snap)
    assert out.pools["a"].tvl == Decimal("50000")
    # APY 不变
    assert out.pools["a"].apy_series == snap.pools["a"].apy_series


def test_apply_does_not_mutate_input():
    inj = EventInjector([
        StressEvent(EventType.GAS_SPIKE, 0, 1, Decimal("1.0")),
        StressEvent(EventType.POOL_EXPLOIT, 0, 1, Decimal("0.9"), target_pool_id="a"),
    ])
    snap = _snap({"a": _pool("a", apy="0.10", tvl="1000000")})
    original_gas = snap.env.gas_base_fee
    original_apy = snap.pools["a"].apy_series[-1]
    original_tvl = snap.pools["a"].tvl

    _ = inj.apply(snap)

    assert snap.env.gas_base_fee == original_gas
    assert snap.pools["a"].apy_series[-1] == original_apy
    assert snap.pools["a"].tvl == original_tvl


def test_apply_returns_new_object():
    inj = EventInjector([
        StressEvent(EventType.GAS_SPIKE, 0, 1, Decimal("1.0")),
    ])
    snap = _snap({"a": _pool("a")})
    out = inj.apply(snap)
    assert out is not snap


def test_apply_unknown_target_pool_logs_and_skips():
    inj = EventInjector([
        StressEvent(EventType.POOL_EXPLOIT, 0, 1, Decimal("0.5"), target_pool_id="missing"),
    ])
    snap = _snap({"a": _pool("a")})
    out = inj.apply(snap)
    # 未崩溃，原池数据不变
    assert out.pools["a"].tvl == snap.pools["a"].tvl


# ----- 调度加载 -----

def test_load_schedule_from_yaml(tmp_path: Path):
    plan = {
        "events": [
            {"event_type": "GAS_SPIKE", "start_tick": 100, "duration": 5, "impact_ratio": "4.0"},
            {"event_type": "POOL_EXPLOIT", "start_tick": 200, "duration": 1,
             "impact_ratio": "0.9", "target_pool_id": "pool_a"},
        ],
    }
    p = tmp_path / "events.yaml"
    p.write_text(yaml.safe_dump(plan), encoding="utf-8")

    inj = EventInjector.load_schedule(p)
    assert len(inj.schedule) == 2
    assert inj.schedule[0].event_type == EventType.GAS_SPIKE
    assert inj.schedule[1].target_pool_id == "pool_a"


def test_load_schedule_rejects_unknown_format(tmp_path: Path):
    p = tmp_path / "events.txt"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        EventInjector.load_schedule(p)


def test_load_schedule_rejects_missing_events_key(tmp_path: Path):
    p = tmp_path / "events.yaml"
    p.write_text(yaml.safe_dump({"foo": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        EventInjector.load_schedule(p)
