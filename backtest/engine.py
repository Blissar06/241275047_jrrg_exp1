"""回测主引擎（FR-06）。

每个 tick 推进顺序：
  1. EventInjector.apply()       事件注入
  2. _accrue_yield()              当前持仓按 APY 计提浮盈
  3. ScoringEngine.run()          打分排序
  4. ReinvestEngine.evaluate()    必要时 commit_reinvest()
  5. RotationEngine.evaluate()    必要时 commit()
  6. NAV / 决策 / 评分日志写入

输出 BacktestResult 含 4 张 DataFrame：
  - nav_log     每 tick 的净值
  - trade_log   每 tick 的轮动决策（ROTATE 与 HOLD 都记录，供 FR-08 归因）
  - reinvest_log
  - score_log   每 tick × 每池的综合得分与各分量

约束（NFR-01/02）：
  - 所有金融运算 Decimal；Parquet 落盘时按需转 float64
  - 同一份 snapshots 多次 run() 结果完全一致
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from data_model.asset import AssetSnapshot
from strategy.interfaces import (
    DecisionType,
    Position,
    ReinvestLog,
    ScoringContext,
    TradeLog,
)
from strategy.reinvest_engine import ReinvestEngine
from strategy.rotation_engine import RotationEngine
from strategy.scoring_engine import ScoringEngine
from backtest.event_injector import EventInjector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    """回测产物。可选地序列化为 Parquet。"""

    nav_log: pd.DataFrame
    trade_log: pd.DataFrame
    reinvest_log: pd.DataFrame
    score_log: pd.DataFrame
    final_position: Position
    snapshots_processed: int

    def persist(self, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # 空 DataFrame 也写：保证文件存在，下游报表无需处理缺失
        self.nav_log.to_parquet(out / "nav_log.parquet", index=False)
        self.trade_log.to_parquet(out / "trade_log.parquet", index=False)
        self.reinvest_log.to_parquet(out / "reinvest_log.parquet", index=False)
        self.score_log.to_parquet(out / "score_log.parquet", index=False)
        logger.info("persisted backtest logs to %s", out)


class BacktestEngine:
    """回测主引擎。"""

    def __init__(
        self,
        initial_capital: Decimal,
        scoring_engine: ScoringEngine,
        rotation_engine: RotationEngine,
        reinvest_engine: ReinvestEngine,
        event_injector: Optional[EventInjector] = None,
        ticks_per_year: int = 365,
        seed: int = 42,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError(f"initial_capital 必须 > 0，got {initial_capital}")
        self.initial_capital = initial_capital
        self.scoring = scoring_engine
        self.rotation = rotation_engine
        self.reinvest = reinvest_engine
        self.events = event_injector or EventInjector()
        self.ticks_per_year = ticks_per_year
        self.seed = seed

    # ---------- 主流程 ----------

    def run(
        self,
        snapshots: Iterable[AssetSnapshot],
        output_dir: Optional[str | Path] = None,
    ) -> BacktestResult:
        position = Position.empty(initial_cash=self.initial_capital)

        nav_records: List[dict] = []
        trade_records: List[dict] = []
        reinvest_records: List[dict] = []
        score_records: List[dict] = []

        n = 0
        prev_snap: Optional[AssetSnapshot] = None
        for raw_snap in snapshots:
            snap = self.events.apply(raw_snap)

            # 0) Mark-to-market：用 (curr/prev) token_price 比例重估持仓
            # 这一步把价格波动传导进 NAV，让回撤/夏普反映真实风险
            if prev_snap is not None:
                position = self._mark_to_market(position, prev_snap, snap)

            # 1) 浮盈累计
            position = self._accrue_yield(position, snap)

            # 2) 评分
            ranking = self.scoring.run(
                snap,
                ScoringContext(current_position=position.pool_id),
            )
            for ps in ranking.rankings:
                pm = snap.pools.get(ps.pool_id)
                # 同时记录原始 APY，便于 Phase 4 attribution 计算理论最优路径
                current_apy = float(pm.apy_series[-1]) if pm and pm.apy_series else 0.0
                score_records.append({
                    "tick": snap.tick,
                    "timestamp": snap.env.timestamp,
                    "pool_id": ps.pool_id,
                    "total_score": float(ps.score),
                    "apy": current_apy,
                    **{k: float(v) for k, v in ps.components.items()},
                })

            # 3) 复投判定
            r_decision = self.reinvest.evaluate(position, snap)
            if r_decision.do_reinvest:
                position, r_log = self.reinvest.commit_reinvest(r_decision, position, snap)
                reinvest_records.append(_reinvest_log_to_dict(r_log))

            # 4) 轮动判定
            decision = self.rotation.evaluate(position, ranking, snap)
            from_pool_pre_commit = position.pool_id
            invested_pre = position.principal + position.cash + position.pending_reward

            if decision.decision_type == DecisionType.ROTATE:
                position, t_log = self.rotation.commit(decision, position, snap)
                trade_records.append(_trade_log_to_dict(t_log))
            else:
                # HOLD 也写入 trade_log，金额为 0，便于 FR-08 归因「调仓空窗折损」
                trade_records.append({
                    "tick": snap.tick,
                    "timestamp": snap.env.timestamp,
                    "operation": "HOLD",
                    "from_pool_id": from_pool_pre_commit,
                    "to_pool_id": decision.target_pool_id,
                    "amount": 0.0,
                    "gas_cost": float(decision.estimated_friction.gas),
                    "slippage_cost": float(decision.estimated_friction.slippage),
                    "lvr_cost": float(decision.estimated_friction.lvr),
                    "expected_gain": float(decision.expected_gain),
                    "decision_reason": decision.reason.value if decision.reason else "OK",
                })

            # 5) NAV 快照（带环境字段，便于事件传播观测 + Phase 4 报表）
            nav_records.append({
                "tick": snap.tick,
                "timestamp": snap.env.timestamp,
                "nav": float(position.total_value()),
                "principal": float(position.principal),
                "pending_reward": float(position.pending_reward),
                "cash": float(position.cash),
                "pool_id": position.pool_id,
                "env_gas_base_fee": float(snap.env.gas_base_fee),
                "env_gas_priority_fee": float(snap.env.gas_priority_fee),
            })

            prev_snap = snap
            n += 1

        result = BacktestResult(
            nav_log=pd.DataFrame(nav_records),
            trade_log=pd.DataFrame(trade_records),
            reinvest_log=pd.DataFrame(reinvest_records),
            score_log=pd.DataFrame(score_records),
            final_position=position,
            snapshots_processed=n,
        )

        if output_dir is not None:
            result.persist(output_dir)

        return result

    # ---------- 内部 ----------

    def _mark_to_market(
        self,
        position: Position,
        prev_snap: AssetSnapshot,
        curr_snap: AssetSnapshot,
    ) -> Position:
        """按持仓池的 token_price 比例重估 principal + pending_reward。

        - 当前未持仓（pool_id None）→ 不动
        - 持仓池在前/后任一 snapshot 中缺失 → 不动（保守）
        - prev_price ≤ 0 → 不动（避免除 0）

        ratio = curr_price / prev_price
        new_principal = principal * ratio
        new_pending  = pending  * ratio
        cash 不变（cash 在计价本位）。
        """
        if position.pool_id is None:
            return position
        if (position.pool_id not in prev_snap.pools
                or position.pool_id not in curr_snap.pools):
            return position
        prev_price = prev_snap.pools[position.pool_id].token_price
        curr_price = curr_snap.pools[position.pool_id].token_price
        if prev_price <= 0:
            return position
        ratio = curr_price / prev_price
        if ratio == Decimal(1):
            return position
        return Position(
            pool_id=position.pool_id,
            principal=position.principal * ratio,
            pending_reward=position.pending_reward * ratio,
            cash=position.cash,
            opened_tick=position.opened_tick,
            last_compound_tick=position.last_compound_tick,
        )

    def _accrue_yield(self, position: Position, snapshot: AssetSnapshot) -> Position:
        """按当前持仓池的最新 APY 线性计提浮盈，归入 pending_reward。

        per_tick_yield = principal × APY_now / ticks_per_year
        """
        if position.pool_id is None or position.pool_id not in snapshot.pools:
            return position
        series = snapshot.pools[position.pool_id].apy_series
        if not series:
            return position
        apy = series[-1]
        if apy == 0 or position.principal == 0:
            return position
        per_tick = position.principal * apy / Decimal(self.ticks_per_year)
        return Position(
            pool_id=position.pool_id,
            principal=position.principal,
            pending_reward=position.pending_reward + per_tick,
            cash=position.cash,
            opened_tick=position.opened_tick,
            last_compound_tick=position.last_compound_tick,
        )


# =================================================================
# 日志记录 → dict 转换（Decimal 转 float，保留 timestamp/Enum 友好类型）
# =================================================================

def _trade_log_to_dict(t: TradeLog) -> dict:
    return {
        "tick": t.tick,
        "timestamp": t.timestamp,
        "operation": t.operation.value,
        "from_pool_id": t.from_pool_id,
        "to_pool_id": t.to_pool_id,
        "amount": float(t.amount),
        "gas_cost": float(t.gas_cost),
        "slippage_cost": float(t.slippage_cost),
        "lvr_cost": float(t.lvr_cost),
        "expected_gain": float(t.expected_gain),
        "decision_reason": t.decision_reason,
    }


def _reinvest_log_to_dict(r: ReinvestLog) -> dict:
    return {
        "tick": r.tick,
        "timestamp": r.timestamp,
        "pool_id": r.pool_id,
        "reward_compounded": float(r.reward_compounded),
        "gas_cost": float(r.gas_cost),
        "expected_gain": float(r.expected_gain),
    }
