"""轮动决策引擎（FR-04 门槛约束型资产轮动）。

决策流水线（evaluate）：
  1. _check_tau_reset：top-1 与当前持仓的 APY 相对偏离 ≤ τ → HOLD(TAU_FAIL)
  2. expected_gain  ← IGainEstimator
  3. friction       ← IFrictionEstimator
  4. _gate：expected_gain ≥ friction.total + threshold × principal → ROTATE，否则 HOLD(GATE_FAIL)

异常处理（E-RT-001/004）：
  - DataIntegrityError（snapshot 字段缺失） → state=ERROR，fallback HOLD(DATA_ERROR)
  - friction 估算返回负值 → 回退上一 tick 缓存值；若无缓存视作 0（保守）
  - 缓存键为 (op_type, pool_id)

commit() 是唯一带副作用的方法：返回新 Position + TradeLog。
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, Optional, Tuple

from data_model.asset import AssetSnapshot
from strategy.interfaces import (
    DataIntegrityError,
    DecisionType,
    FrictionBreakdown,
    HoldReason,
    IFrictionEstimator,
    IGainEstimator,
    OperationType,
    Position,
    RankingTable,
    RotationDecision,
    RotationState,
    TradeLog,
)

logger = logging.getLogger(__name__)


class RotationEngine:
    """门槛约束型轮动引擎。"""

    def __init__(
        self,
        tau_reset: Decimal,
        threshold: Decimal,
        gain_estimator: IGainEstimator,
        friction_estimator: IFrictionEstimator,
        gain_horizon_ticks: int = 30,
    ) -> None:
        if tau_reset < 0:
            raise ValueError(f"tau_reset 必须 >= 0，got {tau_reset}")
        if threshold < 0:
            raise ValueError(f"threshold 必须 >= 0，got {threshold}")
        if gain_horizon_ticks <= 0:
            raise ValueError(f"gain_horizon_ticks 必须为正，got {gain_horizon_ticks}")

        self.tau_reset = tau_reset
        self.threshold = threshold
        self._gain = gain_estimator
        self._friction = friction_estimator
        self.gain_horizon_ticks = gain_horizon_ticks

        self._state: RotationState = RotationState.IDLE
        # 摩擦缓存：(op_type, pool_id) -> 上一 tick 的 FrictionBreakdown
        self._friction_cache: Dict[Tuple[OperationType, str], FrictionBreakdown] = {}
        # 等待 commit 的决策；evaluate ROTATE 后置位
        self._pending: Optional[RotationDecision] = None

    @property
    def state(self) -> RotationState:
        return self._state

    @property
    def pending_decision(self) -> Optional[RotationDecision]:
        return self._pending

    # ---------- 内部辅助 ----------

    def _hold(
        self,
        tick: int,
        target_pool_id: Optional[str],
        reason: HoldReason,
        expected_gain: Decimal = Decimal(0),
        friction: Optional[FrictionBreakdown] = None,
        threshold_required: Decimal = Decimal(0),
        notes: str = "",
    ) -> RotationDecision:
        self._state = RotationState.HOLDING
        self._pending = None
        return RotationDecision(
            tick=tick,
            decision_type=DecisionType.HOLD,
            target_pool_id=target_pool_id,
            expected_gain=expected_gain,
            estimated_friction=friction or FrictionBreakdown(Decimal(0), Decimal(0), Decimal(0)),
            threshold_required=threshold_required,
            reason=reason,
            notes=notes,
        )

    def _check_tau_reset(
        self,
        position: Position,
        ranking: RankingTable,
        snapshot: AssetSnapshot,
    ) -> bool:
        """偏离度检验。spec 对应 |Δscore| ≥ τ 或 |Δyield| ≥ τ_y，任一条件成立即通过。

        - 当前未持仓：直接通过（必须开仓）
        - 当前池在 snapshot 中消失：视作"被强制退出"，通过
        - 检查 1：top-1 与当前持仓的综合分差 > τ（z-score 量纲，捕捉价格风险）
        - 检查 2：APY 相对偏离 > τ（兜底，保留原语义）
        """
        if position.pool_id is None:
            return True
        if position.pool_id not in snapshot.pools:
            logger.warning(
                "tick=%d 当前持仓池=%s 在 snapshot 中缺失，强制视作 τ 通过",
                snapshot.env.tick, position.pool_id,
            )
            return True
        if not ranking.rankings:
            return False
        top = ranking.rankings[0]

        # 数据完整性检查：apy_series 缺失 → DataIntegrityError，由 evaluate() 兜底
        cur_series = snapshot.pools[position.pool_id].apy_series
        tgt_series = snapshot.pools[top.pool_id].apy_series
        if not cur_series or not tgt_series:
            raise DataIntegrityError(
                f"apy_series 为空：current={position.pool_id} target={top.pool_id}"
            )

        # 检查 1：综合分差（score 是 z-score，τ 直接作为绝对差阈值）
        current_in_ranking = ranking.get(position.pool_id)
        if current_in_ranking is not None:
            score_gap = top.score - current_in_ranking.score
            if score_gap > self.tau_reset:
                return True

        # 检查 2：APY 相对偏离（兜底）
        cur_apy = cur_series[-1]
        tgt_apy = tgt_series[-1]
        if abs(cur_apy) < Decimal("1E-9"):
            return tgt_apy > Decimal(0)
        relative = (tgt_apy - cur_apy) / abs(cur_apy)
        return relative > self.tau_reset

    def _safe_friction(
        self,
        op_type: OperationType,
        amount: Decimal,
        pool_id: str,
        snapshot: AssetSnapshot,
    ) -> FrictionBreakdown:
        """调用估算器；若返回任一负分量，回退缓存（无缓存则置 0）。"""
        try:
            fb = self._friction.estimate(op_type, amount, pool_id, snapshot)
        except Exception as e:
            logger.warning(
                "tick=%d friction estimate failed for op=%s pool=%s: %s; falling back to cache",
                snapshot.env.tick, op_type.value, pool_id, e,
            )
            fb = self._friction_cache.get(
                (op_type, pool_id),
                FrictionBreakdown(Decimal(0), Decimal(0), Decimal(0)),
            )
            return fb

        if fb.gas < 0 or fb.slippage < 0 or fb.lvr < 0:
            cached = self._friction_cache.get((op_type, pool_id))
            logger.warning(
                "tick=%d friction returned negative for op=%s pool=%s: %s; falling back to %s",
                snapshot.env.tick, op_type.value, pool_id, fb, cached,
            )
            fb = cached or FrictionBreakdown(Decimal(0), Decimal(0), Decimal(0))
        else:
            self._friction_cache[(op_type, pool_id)] = fb
        return fb

    def _gate(
        self,
        expected_gain: Decimal,
        friction_total: Decimal,
        principal: Decimal,
    ) -> Tuple[bool, Decimal]:
        """门槛函数：expected_gain ≥ friction + threshold × principal。

        threshold 解释为「净增益相对本金的最低占比」，
        principal=0（首次入场）时仍要求 expected_gain ≥ friction。
        """
        threshold_required = self.threshold * principal
        return expected_gain >= friction_total + threshold_required, threshold_required

    # ---------- 主入口 ----------

    def evaluate(
        self,
        position: Position,
        ranking: RankingTable,
        snapshot: AssetSnapshot,
    ) -> RotationDecision:
        """对当前持仓与候选排名表做轮动判定。"""
        tick = snapshot.env.tick
        self._state = RotationState.EVALUATING

        try:
            if not ranking.rankings:
                return self._hold(tick, None, HoldReason.NO_CANDIDATES)

            top = ranking.rankings[0]

            # 同池：无需轮动
            if position.pool_id is not None and top.pool_id == position.pool_id:
                return self._hold(tick, top.pool_id, HoldReason.SAME_POOL)

            # 1) τ 检定（score 或 APY 任一偏离即通过）
            if not self._check_tau_reset(position, ranking, snapshot):
                return self._hold(tick, top.pool_id, HoldReason.TAU_FAIL)

            # 2) 增益估算
            expected_gain = self._gain.expected_rotation_gain(
                position, top.pool_id, snapshot, self.gain_horizon_ticks,
            )

            # 3) 摩擦估算（按"全仓切换"的金额估）
            invested = position.principal + position.cash + position.pending_reward
            friction = self._safe_friction(
                OperationType.ROTATE, invested, top.pool_id, snapshot,
            )

            # 4) 门槛
            principal_basis = invested if invested > 0 else Decimal(0)
            passed, thr_req = self._gate(expected_gain, friction.total, principal_basis)
            if not passed:
                return self._hold(
                    tick, top.pool_id, HoldReason.GATE_FAIL,
                    expected_gain=expected_gain,
                    friction=friction,
                    threshold_required=thr_req,
                )

            # 5) ROTATE
            decision = RotationDecision(
                tick=tick,
                decision_type=DecisionType.ROTATE,
                target_pool_id=top.pool_id,
                expected_gain=expected_gain,
                estimated_friction=friction,
                threshold_required=thr_req,
                reason=None,
            )
            self._state = RotationState.RANKED
            self._pending = decision
            return decision

        except DataIntegrityError as e:
            logger.error("tick=%d DataIntegrityError: %s — fallback HOLD", tick, e)
            decision = self._hold(
                tick,
                ranking.rankings[0].pool_id if ranking.rankings else None,
                HoldReason.DATA_ERROR,
                notes=str(e),
            )
            # _hold() 把 state 设成了 HOLDING；ERROR 路径需要保留 ERROR 状态便于上层观测
            self._state = RotationState.ERROR
            return decision

    def commit(
        self,
        decision: RotationDecision,
        position: Position,
        snapshot: AssetSnapshot,
    ) -> Tuple[Position, TradeLog]:
        """执行 ROTATE 决策。返回 (新仓位, TradeLog)。

        约束：仅 decision_type=ROTATE 可 commit；HOLD 调用此方法抛 ValueError。
        """
        if decision.decision_type != DecisionType.ROTATE:
            raise ValueError(
                f"commit() 仅适用于 ROTATE 决策，got {decision.decision_type}"
            )
        if decision.target_pool_id is None:
            raise ValueError("ROTATE 决策必须指定 target_pool_id")

        self._state = RotationState.COMMITTING
        env = snapshot.env

        invested_pre = position.principal + position.cash + position.pending_reward
        friction = decision.estimated_friction

        # 出仓 + 入场总摩擦从本金中扣除
        new_principal = invested_pre - friction.total
        if new_principal < 0:
            new_principal = Decimal(0)

        new_position = Position(
            pool_id=decision.target_pool_id,
            principal=new_principal,
            pending_reward=Decimal(0),
            cash=Decimal(0),
            opened_tick=env.tick,
            last_compound_tick=env.tick,
        )

        log = TradeLog(
            tick=env.tick,
            timestamp=env.timestamp,
            operation=OperationType.ROTATE,
            from_pool_id=position.pool_id,
            to_pool_id=decision.target_pool_id,
            amount=invested_pre,
            gas_cost=friction.gas,
            slippage_cost=friction.slippage,
            lvr_cost=friction.lvr,
            expected_gain=decision.expected_gain,
            decision_reason="OK",
        )

        self._state = RotationState.IDLE
        self._pending = None
        logger.info(
            "tick=%d ROTATE %s -> %s amount=%s friction=%s",
            env.tick, position.pool_id, decision.target_pool_id,
            invested_pre, friction.total,
        )
        return new_position, log
