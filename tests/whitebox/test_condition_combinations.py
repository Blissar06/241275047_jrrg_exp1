"""白盒：条件组合覆盖测试（Condition Combination Coverage）。

PPT 第 9 讲 §一·5：对单一判定中的多个条件，枚举所有真/假组合。
本文件聚焦项目中含多条件 boolean 的关键判定点。

────────────────────────────────────────────────────────────────────────
判定 #1：FrictionEstimator.estimate 内的负值检测
    if fb.gas < 0 or fb.slippage < 0 or fb.lvr < 0:
    3 个条件 × 2^3 = 8 个组合
────────────────────────────────────────────────────────────────────────
判定 #2：RotationEngine._safe_friction 的回退条件
    fb.gas < 0 or fb.slippage < 0 or fb.lvr < 0
    （和判定 #1 相同结构）
────────────────────────────────────────────────────────────────────────
判定 #3：_check_tau_reset 的逻辑短路
    A: position.pool_id is None
    B: position.pool_id not in snapshot.pools
    C: ranking.rankings 空
    D: cur_series 或 tgt_series 空（→ DataIntegrityError）
    E: score_gap > τ
    F: APY 相对偏离 > τ
    完整真值表过于庞大；选关键无冗余组合。
────────────────────────────────────────────────────────────────────────
判定 #4：_gate
    expected_gain >= friction_total + threshold × principal
    单一比较，但 3 个输入；测条件组合：
    G1: expected_gain 大/小   G2: friction 大/小   G3: threshold 大/小
────────────────────────────────────────────────────────────────────────

编号规范：CC-<判定缩写>-<NN>。
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict

import pytest

from backtest.cost_model import FrictionEstimator
from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import (
    DecisionType,
    FrictionBreakdown,
    HoldReason,
    OperationType,
    PoolScore,
    Position,
    RankingTable,
    RotationState,
)
from strategy.rotation_engine import RotationEngine

pytestmark = [pytest.mark.whitebox, pytest.mark.condition]


# =====================================================================
# 工具
# =====================================================================

class _ConditionalFrictionStub:
    """可逐分量配置的 stub friction estimator，用于注入负值组合。"""

    def __init__(self, gas: Decimal, slippage: Decimal, lvr: Decimal):
        self._fb = FrictionBreakdown(gas=gas, slippage=slippage, lvr=lvr)

    def estimate(self, op_type, amount, pool_id, snapshot) -> FrictionBreakdown:
        return self._fb


def _pool(pid: str, apy: str = "0.05") -> PoolMetrics:
    return PoolMetrics(
        pool_id=pid, apy_series=(Decimal(apy),) * 14,
        tvl=Decimal("100000000"), vol_30d=Decimal("0.02"),
        token_price=Decimal("1.0"), gas_base_fee=Decimal("0.0000001"),
    )


def _snap(pools, tick=0):
    env = EnvSnapshot(
        tick=tick, timestamp=datetime(2024, 1, 1),
        oracle_price={pid: Decimal("1.0") for pid in pools},
        gas_base_fee=Decimal("0.0000001"),
        gas_priority_fee=Decimal("0.00000005"),
    )
    return AssetSnapshot(tick=tick, pools=pools, env=env)


def _rank(scores):
    return RankingTable(
        snapshot_tick=0,
        rankings=tuple(
            PoolScore(pool_id=pid, score=Decimal(s), components={"momentum": Decimal(s)})
            for pid, s in scores
        ),
    )


# =====================================================================
# 判定 #1 / #2：负值检测 3 条件 × 2³ = 8 组合
# 通过 RotationEngine._safe_friction 路径（间接调 estimate + 检查负值）
# =====================================================================
#
# A = gas < 0,   B = slippage < 0,   C = lvr < 0
# 期望：
#   - ABC 全 False → 正常缓存
#   - ABC 任一 True → 回退
#
# 矩阵：(A, B, C)
#   0 (F,F,F) → 正常 (不回退)                CC-NEG-01
#   1 (F,F,T)                                  CC-NEG-02
#   2 (F,T,F)                                  CC-NEG-03
#   3 (F,T,T)                                  CC-NEG-04
#   4 (T,F,F)                                  CC-NEG-05
#   5 (T,F,T)                                  CC-NEG-06
#   6 (T,T,F)                                  CC-NEG-07
#   7 (T,T,T)                                  CC-NEG-08

class TestFrictionNegativeDetection:

    def _engine_with_friction(self, fb_stub) -> RotationEngine:
        return RotationEngine(
            tau_reset=Decimal("0.001"),
            threshold=Decimal("0.0001"),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            friction_estimator=fb_stub,
            gain_horizon_ticks=30,
        )

    @pytest.mark.parametrize("idx,gas,slip,lvr,should_fall_back", [
        # idx, gas, slip, lvr, expect_fallback (True=回退 / False=正常使用)
        (1, Decimal("10"),   Decimal("5"),   Decimal("1"),   False),  # F F F
        (2, Decimal("10"),   Decimal("5"),   Decimal("-1"),  True),   # F F T
        (3, Decimal("10"),   Decimal("-1"),  Decimal("1"),   True),   # F T F
        (4, Decimal("10"),   Decimal("-1"),  Decimal("-1"),  True),   # F T T
        (5, Decimal("-1"),   Decimal("5"),   Decimal("1"),   True),   # T F F
        (6, Decimal("-1"),   Decimal("5"),   Decimal("-1"),  True),   # T F T
        (7, Decimal("-1"),   Decimal("-1"),  Decimal("1"),   True),   # T T F
        (8, Decimal("-1"),   Decimal("-1"),  Decimal("-1"),  True),   # T T T
    ])
    def test_CC_NEG_combinations(self, idx, gas, slip, lvr, should_fall_back):
        """8 个 (gas<0, slip<0, lvr<0) 组合：只要任一为真，应回退到 0/cache。"""
        stub = _ConditionalFrictionStub(gas=gas, slippage=slip, lvr=lvr)
        engine = self._engine_with_friction(stub)
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.50")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        if should_fall_back:
            # 回退到 0 → friction.total = 0
            assert d.estimated_friction.total == Decimal(0), (
                f"CC-NEG-0{idx}: 期望回退到 0，实际 friction.total={d.estimated_friction.total}"
            )
        else:
            assert d.estimated_friction.total > Decimal(0), (
                f"CC-NEG-0{idx}: 期望正常 friction，实际为 0"
            )


# =====================================================================
# 判定 #3：_check_tau_reset 的多条件短路
# 选 5 个关键组合（不穷举：6 条件 × 2^6 = 64 个不实际）
# =====================================================================
#
# A: 无持仓     B: 池缺失   C: ranking 空
# D: apy 序列空 E: score 差大 F: APY 偏离大
#
# 重要组合：
#   CC-TAU-01: A=T (其它无关) → True
#   CC-TAU-02: A=F, B=T → True (warning)
#   CC-TAU-03: A=F, B=F, C=T → False (NO_CANDIDATES 由 evaluate 拦截)
#   CC-TAU-04: A=F, B=F, C=F, D=T → DataIntegrityError
#   CC-TAU-05: A=F, B=F, C=F, D=F, E=T, F=F → True (score 路径)
#   CC-TAU-06: A=F, B=F, C=F, D=F, E=F, F=T → True (APY 路径)
#   CC-TAU-07: A=F, B=F, C=F, D=F, E=F, F=F → False (TAU_FAIL)

class TestCheckTauResetCombinations:

    def _engine(self, tau: str = "0.05") -> RotationEngine:
        from tests._stubs import StubFrictionEstimator
        return RotationEngine(
            tau_reset=Decimal(tau), threshold=Decimal("0.001"),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            friction_estimator=StubFrictionEstimator(gas=Decimal("10")),
            gain_horizon_ticks=30,
        )

    def test_CC_TAU_01_no_position(self):
        engine = self._engine()
        pools = {"a": _pool("a"), "b": _pool("b")}
        pos = Position.empty(initial_cash=Decimal("100000"))
        d = engine.evaluate(pos, _rank([("a", "1.0"), ("b", "0.5")]), _snap(pools))
        assert d.reason != HoldReason.TAU_FAIL

    def test_CC_TAU_02_current_pool_missing(self):
        engine = self._engine()
        pools = {"b": _pool("b")}  # 没 "a"
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0")]), _snap(pools))
        # 应进入 gate（不报 TAU_FAIL）
        assert d.reason != HoldReason.TAU_FAIL

    def test_CC_TAU_03_empty_ranking_handled_by_evaluate(self):
        engine = self._engine()
        pools = {"a": _pool("a")}
        pos = Position.empty(initial_cash=Decimal("100000"))
        d = engine.evaluate(pos, _rank([]), _snap(pools))
        assert d.reason == HoldReason.NO_CANDIDATES

    def test_CC_TAU_04_empty_apy_series_raises_data_error(self):
        engine = self._engine()
        a = PoolMetrics(
            pool_id="a", apy_series=(),
            tvl=Decimal("100000000"), vol_30d=Decimal("0.02"),
            token_price=Decimal("1.0"), gas_base_fee=Decimal("0.0000001"),
        )
        pools = {"a": a, "b": _pool("b", "0.20")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.reason == HoldReason.DATA_ERROR

    def test_CC_TAU_05_score_gap_passes(self):
        """E=T, F=F：score 差大、APY 差小 → 走 score 通过分支。"""
        engine = self._engine(tau="0.05")
        # APY 几乎相等
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.051")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        # score 差 1.5 > τ=0.05
        d = engine.evaluate(pos, _rank([("b", "2.0"), ("a", "0.5")]), _snap(pools))
        assert d.reason != HoldReason.TAU_FAIL

    def test_CC_TAU_06_apy_relative_passes(self):
        """E=F, F=T：score 差小，APY 偏离大 → 走 APY 通过分支。"""
        engine = self._engine(tau="0.05")
        # APY 差 +400%（远大于 τ）
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.30")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        # score 差小
        d = engine.evaluate(pos, _rank([("b", "0.51"), ("a", "0.50")]), _snap(pools))
        assert d.reason != HoldReason.TAU_FAIL

    def test_CC_TAU_07_both_fail(self):
        """E=F, F=F：双双失败 → TAU_FAIL。"""
        engine = self._engine(tau="0.10")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.052")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "0.51"), ("a", "0.50")]), _snap(pools))
        assert d.reason == HoldReason.TAU_FAIL


# =====================================================================
# 判定 #4：gate 比较 3 因素组合 (gain, friction, threshold)
#
# CC-GATE 矩阵：每个因素取 大(H) / 小(L)
#   (gain_H, friction_L, thr_L) → pass        CC-GATE-01
#   (gain_H, friction_H, thr_L) → may pass/fail
#   (gain_H, friction_L, thr_H) → may pass/fail
#   (gain_L, friction_L, thr_L) → fail        CC-GATE-04
#   ...
# 选 4 个有意义的组合
# =====================================================================

class TestGateCombinations:

    def _engine(self, threshold: str, gas: str) -> RotationEngine:
        from tests._stubs import StubFrictionEstimator
        return RotationEngine(
            tau_reset=Decimal("0.01"), threshold=Decimal(threshold),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            friction_estimator=StubFrictionEstimator(gas=Decimal(gas)),
            gain_horizon_ticks=365,  # 长 horizon → gain 容易大
        )

    def test_CC_GATE_01_high_gain_low_friction_low_threshold_passes(self):
        engine = self._engine(threshold="0.0001", gas="1")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.50")}  # 高 gain
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.decision_type == DecisionType.ROTATE

    def test_CC_GATE_02_high_gain_high_friction_low_threshold_fails(self):
        engine = self._engine(threshold="0.0001", gas="80000")  # 极高 gas
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.30")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.reason == HoldReason.GATE_FAIL

    def test_CC_GATE_03_high_gain_low_friction_high_threshold_may_fail(self):
        engine = self._engine(threshold="0.05", gas="1")  # 阈值 5%×100k = 5000
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.08")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        # gain ≈ 100k × 0.03 = 3000 < threshold 5000 → fail
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.reason == HoldReason.GATE_FAIL

    def test_CC_GATE_04_low_gain_fails(self):
        engine = self._engine(threshold="0.001", gas="1")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.0501")}  # 几乎相同
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        # gain ≈ 10 < threshold 100 → fail
        assert d.reason == HoldReason.GATE_FAIL
