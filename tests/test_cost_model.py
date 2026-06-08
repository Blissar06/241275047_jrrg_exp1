"""Phase 3 命令 3-1：FrictionEstimator 单元测试。"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict

import pytest

from backtest.cost_model import FrictionEstimator
from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from strategy.interfaces import FrictionBreakdown, OperationType


def _pool(pid: str, tvl: str = "1000000", token_price: str = "1.0") -> PoolMetrics:
    return PoolMetrics(
        pool_id=pid,
        apy_series=(Decimal("0.05"),) * 5,
        tvl=Decimal(tvl),
        vol_30d=Decimal("0.02"),
        token_price=Decimal(token_price),
        gas_base_fee=Decimal("0.0000001"),
    )


def _snap(
    pools: Dict[str, PoolMetrics],
    base_fee: str = "0.0000001",
    prio_fee: str = "0.00000005",
    tick: int = 0,
    oracle_overrides: Dict[str, Decimal] | None = None,
) -> AssetSnapshot:
    oracle = {pid: pools[pid].token_price for pid in pools}
    if oracle_overrides:
        oracle.update(oracle_overrides)
    env = EnvSnapshot(
        tick=tick,
        timestamp=datetime(2024, 1, 1),
        oracle_price=oracle,
        gas_base_fee=Decimal(base_fee),
        gas_priority_fee=Decimal(prio_fee),
    )
    return AssetSnapshot(tick=tick, pools=pools, env=env)


# ----- gas -----

def test_gas_cost_uses_op_specific_gas_limit():
    fe = FrictionEstimator()
    env = EnvSnapshot(
        tick=0, timestamp=datetime(2024, 1, 1), oracle_price={},
        gas_base_fee=Decimal("0.0000001"),
        gas_priority_fee=Decimal("0.00000005"),
    )
    rotate = fe.estimate_gas(OperationType.ROTATE, env)
    reinvest = fe.estimate_gas(OperationType.REINVEST, env)
    # ROTATE gas_limit=350k > REINVEST 180k
    assert rotate > reinvest
    # 数值校验：(1e-7 + 5e-8) * 350000 = 0.0525
    assert rotate == Decimal("0.0000001") * Decimal(350_000) + Decimal("0.00000005") * Decimal(350_000)


# ----- slippage -----

@pytest.mark.parametrize("trade_pct,expected_rate", [
    ("0.005", "0.001"),   # < 1% → low
    ("0.03", "0.003"),    # 1~5% → mid
    ("0.10", "0.008"),    # > 5% → high
])
def test_slippage_step_function(trade_pct, expected_rate):
    fe = FrictionEstimator()
    tvl = Decimal("1000000")
    trade_size = tvl * Decimal(trade_pct)
    cost = fe.estimate_slippage(trade_size, tvl)
    assert cost == trade_size * Decimal(expected_rate)


def test_slippage_zero_tvl_uses_high_rate():
    fe = FrictionEstimator()
    cost = fe.estimate_slippage(Decimal("100"), Decimal(0))
    assert cost == Decimal("100") * Decimal("0.008")


def test_slippage_zero_trade_returns_zero():
    fe = FrictionEstimator()
    assert fe.estimate_slippage(Decimal(0), Decimal("1000000")) == Decimal(0)


# ----- LVR -----

def test_lvr_zero_when_oracle_equals_pool_price():
    fe = FrictionEstimator()
    out = fe.estimate_lvr(
        oracle_price=Decimal("1.0"),
        pool_price=Decimal("1.0"),
        trade_size=Decimal("10000"),
    )
    assert out == Decimal(0)


def test_lvr_nonzero_with_price_divergence():
    fe = FrictionEstimator()
    out = fe.estimate_lvr(
        oracle_price=Decimal("1.0"),
        pool_price=Decimal("1.02"),    # 2% 高估
        trade_size=Decimal("10000"),
    )
    # |1 - 1.02| / 1 * 10000 * 0.5 = 100
    assert out == Decimal("100")


# ----- estimate() integration -----

def test_estimate_rotate_returns_all_three_components():
    fe = FrictionEstimator()
    pools = {"a": _pool("a", tvl="1000000")}
    snap = _snap(pools, oracle_overrides={"a": Decimal("0.95")})
    fb = fe.estimate(OperationType.ROTATE, Decimal("50000"), "a", snap)
    assert fb.gas > 0
    assert fb.slippage > 0
    assert fb.lvr > 0


def test_estimate_reinvest_only_gas():
    fe = FrictionEstimator()
    pools = {"a": _pool("a", tvl="1000000")}
    snap = _snap(pools)
    fb = fe.estimate(OperationType.REINVEST, Decimal("100"), "a", snap)
    assert fb.gas > 0
    assert fb.slippage == Decimal(0)
    assert fb.lvr == Decimal(0)


def test_estimate_unknown_pool_falls_back_to_zero_when_no_cache():
    fe = FrictionEstimator()
    pools = {"a": _pool("a")}
    snap = _snap(pools)
    fb = fe.estimate(OperationType.ROTATE, Decimal("100"), "missing", snap)
    assert fb.total == Decimal(0)


def test_cache_used_after_first_successful_call_then_failure():
    """先成功一次建立缓存，再让池消失，回退到缓存值。"""
    fe = FrictionEstimator()
    pools = {"a": _pool("a")}
    snap1 = _snap(pools, tick=0)
    fb1 = fe.estimate(OperationType.ROTATE, Decimal("50000"), "a", snap1)

    # 下一 tick 池消失
    empty_snap = _snap({}, tick=1)
    fb2 = fe.estimate(OperationType.ROTATE, Decimal("50000"), "a", empty_snap)
    assert fb2 == fb1


# ----- from_config -----

def test_from_config_overrides_defaults():
    cfg = {
        "gas_limits": {"ROTATE": 999_999, "REINVEST": 100_000},
        "slippage_steps": {"low": "0.0005", "mid": "0.005", "high": "0.02"},
    }
    fe = FrictionEstimator.from_config(cfg)
    assert fe.gas_limits[OperationType.ROTATE] == 999_999
    assert fe.slip_rate_low == Decimal("0.0005")
    assert fe.slip_rate_high == Decimal("0.02")


def test_from_config_ignores_unknown_op_type():
    cfg = {"gas_limits": {"ROTATE": 1, "BOGUS_OP": 999}}
    fe = FrictionEstimator.from_config(cfg)
    assert fe.gas_limits[OperationType.ROTATE] == 1
