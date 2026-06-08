"""复投决策引擎（FR-05 净效用驱动自动复投）。

evaluate() 流程：
  1. 若无持仓或 pending_reward<=0 → do_reinvest=False
  2. 估算 expected_gain（IGainEstimator.expected_reinvest_gain）
  3. 估算 gas_cost（IFrictionEstimator.estimate(REINVEST, …)）
  4. 触发条件：expected_gain > gas_cost × risk_premium_multiplier

commit_reinvest() 是唯一带副作用的方法：
  - 从 position.cash 或 pending_reward 中扣除 gas_cost
  - 把 pending_reward 合入 principal
  - 返回 (新 Position, ReinvestLog)
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Tuple

from data_model.asset import AssetSnapshot
from strategy.interfaces import (
    IFrictionEstimator,
    IGainEstimator,
    OperationType,
    Position,
    ReinvestDecision,
    ReinvestLog,
)

logger = logging.getLogger(__name__)


class ReinvestEngine:
    """净效用驱动复投。"""

    def __init__(
        self,
        friction_estimator: IFrictionEstimator,
        gain_estimator: IGainEstimator,
        reinvest_window: int = 30,
        risk_premium_multiplier: Decimal = Decimal("1.5"),
    ) -> None:
        if reinvest_window <= 0:
            raise ValueError(f"reinvest_window 必须为正，got {reinvest_window}")
        if risk_premium_multiplier <= 0:
            raise ValueError(
                f"risk_premium_multiplier 必须为正，got {risk_premium_multiplier}"
            )
        self._friction = friction_estimator
        self._gain = gain_estimator
        self.reinvest_window = reinvest_window
        self.risk_premium_multiplier = risk_premium_multiplier

    # ---------- 主入口 ----------

    def evaluate(
        self,
        position: Position,
        snapshot: AssetSnapshot,
    ) -> ReinvestDecision:
        tick = snapshot.env.tick

        if position.pool_id is None:
            return ReinvestDecision(
                tick=tick, do_reinvest=False,
                pending_reward=Decimal(0), gas_cost=Decimal(0),
                expected_gain=Decimal(0), reason="NO_POSITION",
            )

        if position.pending_reward <= 0:
            return ReinvestDecision(
                tick=tick, do_reinvest=False,
                pending_reward=position.pending_reward, gas_cost=Decimal(0),
                expected_gain=Decimal(0), reason="NO_REWARDS",
            )

        # 复投后未来 window 内的预期增量收益
        expected_gain = self._gain.expected_reinvest_gain(
            position, snapshot, self.reinvest_window,
        )

        # Gas 成本（复投操作；amount 用 pending_reward 估）
        fb = self._friction.estimate(
            OperationType.REINVEST,
            position.pending_reward,
            position.pool_id,
            snapshot,
        )
        gas_cost = fb.total  # 复投时滑点和 LVR 通常 0；以 total 为最坏估计

        threshold_value = gas_cost * self.risk_premium_multiplier
        do_it = expected_gain > threshold_value
        reason = "OK" if do_it else "NEGATIVE_NET"

        return ReinvestDecision(
            tick=tick,
            do_reinvest=do_it,
            pending_reward=position.pending_reward,
            gas_cost=gas_cost,
            expected_gain=expected_gain,
            reason=reason,
        )

    def commit_reinvest(
        self,
        decision: ReinvestDecision,
        position: Position,
        snapshot: AssetSnapshot,
    ) -> Tuple[Position, ReinvestLog]:
        """执行复投。pending_reward 转为 principal；gas_cost 从 cash/pending 中支付。"""
        if not decision.do_reinvest:
            raise ValueError("commit_reinvest() 仅适用于 do_reinvest=True 的决策")
        if position.pool_id is None:
            raise ValueError("无持仓时不可复投")

        env = snapshot.env

        # 支付 gas：优先从 cash 扣，不够再从 pending_reward 扣
        gas_remaining = decision.gas_cost
        new_cash = position.cash
        new_pending = position.pending_reward

        if new_cash >= gas_remaining:
            new_cash -= gas_remaining
            gas_remaining = Decimal(0)
        else:
            gas_remaining -= new_cash
            new_cash = Decimal(0)

        if gas_remaining > 0:
            new_pending -= gas_remaining
            if new_pending < 0:
                # 极端情况：pending 不足以支付剩余 gas，记录但不阻塞
                logger.warning(
                    "tick=%d gas overshoot pending_reward by %s; clipping to 0",
                    env.tick, -new_pending,
                )
                new_pending = Decimal(0)

        compounded = new_pending  # 全部剩余 pending 合入 principal
        new_position = Position(
            pool_id=position.pool_id,
            principal=position.principal + compounded,
            pending_reward=Decimal(0),
            cash=new_cash,
            opened_tick=position.opened_tick,
            last_compound_tick=env.tick,
        )

        log = ReinvestLog(
            tick=env.tick,
            timestamp=env.timestamp,
            pool_id=position.pool_id,
            reward_compounded=compounded,
            gas_cost=decision.gas_cost,
            expected_gain=decision.expected_gain,
        )
        logger.info(
            "tick=%d REINVEST pool=%s compounded=%s gas=%s",
            env.tick, position.pool_id, compounded, decision.gas_cost,
        )
        return new_position, log
