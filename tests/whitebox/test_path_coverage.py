"""白盒：基本路径覆盖测试（Basic Path Coverage）。

PPT 第 9 讲：圈复杂度 V(G) 决定基本路径数；每条基本路径至少 1 个测试用例。
覆盖目标：3 个最关键的高分支密度函数：

  1. RotationEngine.evaluate           (V(G) ≈ 7)
  2. FrictionEstimator.estimate         (V(G) ≈ 5)
  3. RotationEngine._check_tau_reset    (V(G) ≈ 6)
  4. ReinvestEngine.commit_reinvest     (V(G) ≈ 4)

编号规范：PATH-<函数缩写>-<NN>。
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
    ReinvestDecision,
    RotationState,
)
from strategy.reinvest_engine import ReinvestEngine
from strategy.rotation_engine import RotationEngine
from tests._stubs import StubFrictionEstimator

pytestmark = [pytest.mark.whitebox, pytest.mark.path]


# =====================================================================
# 工厂
# =====================================================================

def _pool(pid: str, apy: str = "0.05", empty: bool = False) -> PoolMetrics:
    return PoolMetrics(
        pool_id=pid,
        apy_series=() if empty else (Decimal(apy),) * 14,
        tvl=Decimal("100000000"),
        vol_30d=Decimal("0.02"),
        token_price=Decimal("1.0"),
        gas_base_fee=Decimal("0.0000001"),
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
# 函数 1：RotationEngine.evaluate
# 基本路径分支图（简化）：
#
#   evaluate(...)
#     ├─ ranking 空 → return HOLD(NO_CANDIDATES)             ← PATH-RE-01
#     ├─ top.pool_id == position.pool_id → HOLD(SAME_POOL)   ← PATH-RE-02
#     ├─ _check_tau_reset = False → HOLD(TAU_FAIL)          ← PATH-RE-03
#     ├─ expected_gain 计算 + friction 计算
#     │   ├─ gate fail → HOLD(GATE_FAIL)                    ← PATH-RE-04
#     │   └─ gate pass → ROTATE                             ← PATH-RE-05
#     └─ DataIntegrityError 抛出 → state=ERROR + HOLD       ← PATH-RE-06
# =====================================================================

class TestRotationEvaluatePaths:

    def _engine(self, **kw) -> RotationEngine:
        return RotationEngine(
            tau_reset=Decimal(kw.get("tau", "0.05")),
            threshold=Decimal(kw.get("threshold", "0.001")),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            friction_estimator=StubFrictionEstimator(gas=Decimal(kw.get("gas", "10"))),
            gain_horizon_ticks=30,
        )

    def test_PATH_RE_01_empty_ranking(self):
        """路径 1：ranking 为空 → 直接 HOLD(NO_CANDIDATES)，不进入任何后续判定。"""
        engine = self._engine()
        d = engine.evaluate(
            Position.empty(initial_cash=Decimal("100000")),
            _rank([]),
            _snap({"a": _pool("a")}),
        )
        assert d.decision_type == DecisionType.HOLD
        assert d.reason == HoldReason.NO_CANDIDATES

    def test_PATH_RE_02_same_pool_short_circuit(self):
        """路径 2：top.pool_id == 当前持仓 → HOLD(SAME_POOL)，不进入 τ/gate。"""
        engine = self._engine()
        pools = {"a": _pool("a", "0.10"), "b": _pool("b", "0.05")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("a", "1.0"), ("b", "0.5")]), _snap(pools))
        assert d.reason == HoldReason.SAME_POOL

    def test_PATH_RE_03_tau_fail(self):
        engine = self._engine(tau="0.10")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.052")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "0.51"), ("a", "0.50")]), _snap(pools))
        assert d.reason == HoldReason.TAU_FAIL

    def test_PATH_RE_04_gate_fail(self):
        engine = self._engine(threshold="0.001", gas="50000")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.50")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.reason == HoldReason.GATE_FAIL

    def test_PATH_RE_05_rotate_full_path(self):
        engine = self._engine(threshold="0.0001", gas="10")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.50")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.decision_type == DecisionType.ROTATE
        assert engine.state == RotationState.RANKED  # 等待 commit

    def test_PATH_RE_06_data_integrity_error_path(self):
        engine = self._engine()
        pools = {
            "a": _pool("a", empty=True),
            "b": _pool("b", "0.10"),
        }
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.reason == HoldReason.DATA_ERROR
        assert engine.state == RotationState.ERROR


# =====================================================================
# 函数 2：FrictionEstimator.estimate
# 路径：
#   estimate(...)
#     ├─ pool_id not in snap.pools → DataIntegrityError → cached/zero  ← PATH-FE-01
#     ├─ op_type ∈ {REINVEST, CLAIM} → 仅 gas，slippage=lvr=0           ← PATH-FE-02
#     ├─ op_type ROTATE/DEPOSIT
#     │   ├─ 任一分量负 → 回退 cache                                    ← PATH-FE-03
#     │   └─ 正常 → 缓存 + 返回                                          ← PATH-FE-04
#     └─ 内部抛异常 → cache fallback                                     ← PATH-FE-05
# =====================================================================

class TestFrictionEstimatePaths:

    def test_PATH_FE_01_unknown_pool_falls_back_to_zero(self):
        fe = FrictionEstimator()
        snap = _snap({"a": _pool("a")})
        out = fe.estimate(OperationType.ROTATE, Decimal("100"), "missing", snap)
        assert out.total == Decimal(0)

    def test_PATH_FE_02_reinvest_op_skips_slippage_and_lvr(self):
        fe = FrictionEstimator()
        snap = _snap({"a": _pool("a")})
        out = fe.estimate(OperationType.REINVEST, Decimal("100"), "a", snap)
        assert out.slippage == Decimal(0)
        assert out.lvr == Decimal(0)
        assert out.gas > Decimal(0)

    def test_PATH_FE_03_negative_component_falls_back_to_cache(self):
        """场景：先成功估算建立缓存，再让池消失 → 拉缓存。"""
        fe = FrictionEstimator()
        pools = {"a": _pool("a")}
        snap1 = _snap(pools, tick=0)
        ok = fe.estimate(OperationType.ROTATE, Decimal("50000"), "a", snap1)

        # 让 snap2 不含 "a" pool → 走 fallback 路径
        snap2 = _snap({}, tick=1)
        fb = fe.estimate(OperationType.ROTATE, Decimal("50000"), "a", snap2)
        assert fb == ok   # 缓存被复用

    def test_PATH_FE_04_normal_rotate_returns_three_components(self):
        fe = FrictionEstimator()
        pools = {"a": _pool("a")}
        snap = _snap(pools)
        out = fe.estimate(OperationType.ROTATE, Decimal("50000"), "a", snap)
        # 大 TVL → low 档；slippage > 0；gas > 0；LVR 因 oracle = pool_price = 0
        assert out.gas > Decimal(0)
        assert out.slippage > Decimal(0)
        assert out.lvr == Decimal(0)

    def test_PATH_FE_05_unknown_op_type_uses_default_gas_limit(self):
        """OperationType.DEPOSIT 默认 gas_limit；走正常路径不抛错。"""
        fe = FrictionEstimator()
        snap = _snap({"a": _pool("a")})
        out = fe.estimate(OperationType.DEPOSIT, Decimal("1000"), "a", snap)
        assert out.gas > Decimal(0)


# =====================================================================
# 函数 3：RotationEngine._check_tau_reset
# 路径（通过 evaluate 黑盒触发）：
#   ├─ position.pool_id is None → return True              ← PATH-TR-01
#   ├─ pool_id not in snap.pools → return True (warning)   ← PATH-TR-02
#   ├─ ranking 为空 → return False                          ← PATH-TR-03（不可达；evaluate 提前拦截）
#   ├─ apy_series 空 → raise DataIntegrityError            ← PATH-TR-04
#   ├─ score_gap > τ → return True                          ← PATH-TR-05
#   ├─ |cur_apy| < 1e-9 → 看 tgt_apy 正负                   ← PATH-TR-06
#   └─ APY 相对偏离 > τ → return True；否则 False           ← PATH-TR-07
# =====================================================================

class TestCheckTauResetPaths:

    def _engine(self, tau: str = "0.05") -> RotationEngine:
        return RotationEngine(
            tau_reset=Decimal(tau), threshold=Decimal("0.001"),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            friction_estimator=StubFrictionEstimator(gas=Decimal("10")),
            gain_horizon_ticks=30,
        )

    def test_PATH_TR_01_no_position_passes_tau(self):
        engine = self._engine()
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.06")}
        pos = Position.empty(initial_cash=Decimal("100000"))
        # 无持仓 → τ 直接通过 → 进入 gate（apy diff 1% × 100000 × 30/365 ≈ 82 < gate）
        # 但具体结果不重要，重点是不报 TAU_FAIL
        d = engine.evaluate(pos, _rank([("a", "1.0"), ("b", "0.5")]), _snap(pools))
        assert d.reason != HoldReason.TAU_FAIL

    def test_PATH_TR_02_current_pool_missing_in_snap_passes_tau(self):
        """当前持仓在 snap.pools 缺失 → τ 强制通过（warning 日志）。"""
        engine = self._engine()
        pools = {"b": _pool("b", "0.50")}  # 没有 "a"
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0")]), _snap(pools))
        # τ_reset 通过，会进入 gate；只要不报 TAU_FAIL
        assert d.reason != HoldReason.TAU_FAIL

    def test_PATH_TR_04_empty_apy_series_raises_data_integrity(self):
        engine = self._engine()
        pools = {
            "a": _pool("a", empty=True),
            "b": _pool("b", "0.10"),
        }
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "1.0"), ("a", "0.5")]), _snap(pools))
        assert d.reason == HoldReason.DATA_ERROR

    def test_PATH_TR_05_score_gap_passes_tau(self):
        """score 差 > τ → τ 通过（即使 APY 差小也行）。"""
        engine = self._engine(tau="0.05")
        # APY 几乎相同；但人为 score 差大
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.051")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "2.0"), ("a", "0.5")]), _snap(pools))
        # score gap = 1.5 > 0.05 → τ 通过；可能落到 gate；只要不是 TAU_FAIL
        assert d.reason != HoldReason.TAU_FAIL

    def test_PATH_TR_06_zero_current_apy_branch(self):
        """current APY 接近 0 → 走特殊分支：tgt > 0 即可。"""
        engine = self._engine()
        # 构造 current_apy = 0 的池（series 末位为 0）
        a_pool = PoolMetrics(
            pool_id="a", apy_series=(Decimal(0),) * 14,
            tvl=Decimal("100000000"), vol_30d=Decimal("0.02"),
            token_price=Decimal("1.0"), gas_base_fee=Decimal("0.0000001"),
        )
        pools = {"a": a_pool, "b": _pool("b", "0.10")}
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "0.51"), ("a", "0.50")]), _snap(pools))
        # cur ≈ 0 且 tgt > 0 → τ 通过；走到 gate
        assert d.reason != HoldReason.TAU_FAIL

    def test_PATH_TR_07_apy_relative_within_tau_holds(self):
        """score 差小 + APY 相对偏离 < τ → 双双失败 → TAU_FAIL。"""
        engine = self._engine(tau="0.10")
        pools = {"a": _pool("a", "0.05"), "b": _pool("b", "0.052")}  # APY 仅差 4%
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, _rank([("b", "0.51"), ("a", "0.50")]), _snap(pools))
        assert d.reason == HoldReason.TAU_FAIL


# =====================================================================
# 函数 4：ReinvestEngine.commit_reinvest
# 路径：
#   commit_reinvest
#     ├─ do_reinvest=False → raise ValueError                ← PATH-RI-01
#     ├─ position.pool_id is None → raise ValueError         ← PATH-RI-02
#     ├─ cash >= gas → 从 cash 扣 gas                          ← PATH-RI-03
#     ├─ cash < gas → cash 扣完 + pending 补差                  ← PATH-RI-04
#     └─ pending 不足以补差 → 余 = 0 + WARN                     ← PATH-RI-05
# =====================================================================

class TestReinvestCommitPaths:

    @pytest.fixture
    def engine(self):
        return ReinvestEngine(
            friction_estimator=StubFrictionEstimator(gas=Decimal("1")),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            reinvest_window=180,
            risk_premium_multiplier=Decimal("1.5"),
        )

    def test_PATH_RI_01_no_reinvest_decision_raises(self, engine):
        snap = _snap({"a": _pool("a")})
        d = ReinvestDecision(
            tick=0, do_reinvest=False, pending_reward=Decimal(0),
            gas_cost=Decimal(0), expected_gain=Decimal(0), reason="NO_POSITION",
        )
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal(0), cash=Decimal(0),
                       opened_tick=0, last_compound_tick=0)
        with pytest.raises(ValueError):
            engine.commit_reinvest(d, pos, snap)

    def test_PATH_RI_02_no_position_raises(self, engine):
        snap = _snap({"a": _pool("a")})
        d = ReinvestDecision(
            tick=0, do_reinvest=True, pending_reward=Decimal("100"),
            gas_cost=Decimal("1"), expected_gain=Decimal("50"), reason="OK",
        )
        pos = Position.empty(initial_cash=Decimal("100000"))
        with pytest.raises(ValueError):
            engine.commit_reinvest(d, pos, snap)

    def test_PATH_RI_03_gas_paid_entirely_from_cash(self, engine):
        snap = _snap({"a": _pool("a", "0.20")})
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal("100"), cash=Decimal("50"),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, snap)
        new_pos, _ = engine.commit_reinvest(d, pos, snap)
        # gas=1 ≤ cash=50 → cash=49, pending 全部入本金
        assert new_pos.cash == Decimal("49")
        assert new_pos.principal == Decimal("100100")
        assert new_pos.pending_reward == Decimal(0)

    def test_PATH_RI_04_gas_partly_from_pending(self):
        """cash < gas → cash 扣空 + pending 补差。"""
        engine = ReinvestEngine(
            friction_estimator=StubFrictionEstimator(gas=Decimal("60")),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            reinvest_window=180, risk_premium_multiplier=Decimal("1.5"),
        )
        snap = _snap({"a": _pool("a", "0.20")})
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal("1000"), cash=Decimal("10"),
                       opened_tick=0, last_compound_tick=0)
        d = engine.evaluate(pos, snap)
        new_pos, _ = engine.commit_reinvest(d, pos, snap)
        # cash 0；pending=1000 -50（补差）= 950 注入本金
        assert new_pos.cash == Decimal(0)
        assert new_pos.principal == Decimal("100950")

    def test_PATH_RI_05_pending_insufficient_clipped_to_zero(self):
        """pending 不足以补 gas 差额 → 截到 0，WARN 日志。"""
        engine = ReinvestEngine(
            friction_estimator=StubFrictionEstimator(gas=Decimal("100")),
            gain_estimator=APYDeltaGainEstimator(use_price_drift=False),
            reinvest_window=180, risk_premium_multiplier=Decimal("1.5"),
        )
        snap = _snap({"a": _pool("a", "0.20")})
        # cash=10, pending=20, gas=100 → 总余 30 < 100 → 都耗光
        pos = Position(pool_id="a", principal=Decimal("100000"),
                       pending_reward=Decimal("20"), cash=Decimal("10"),
                       opened_tick=0, last_compound_tick=0)
        # 强制 do_reinvest=True
        d = ReinvestDecision(
            tick=0, do_reinvest=True, pending_reward=Decimal("20"),
            gas_cost=Decimal("100"), expected_gain=Decimal("500"), reason="OK",
        )
        new_pos, _ = engine.commit_reinvest(d, pos, snap)
        # cash 0, pending 0（合入本金 0），principal 不变
        assert new_pos.cash == Decimal(0)
        assert new_pos.pending_reward == Decimal(0)
        assert new_pos.principal == Decimal("100000")
