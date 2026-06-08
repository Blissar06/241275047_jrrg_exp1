"""ScoringEngine：聚合多个 Scorer，按 WeightConfig 加权后排序。

聚合公式：
    Score(pool) = Σ_i w_i · ScoreVector_i.scores[pool]

排序规则：
    主键：score 降序
    副键：pool_id 字典升序（同分稳定）  —— 满足 NFR-02 复现性
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, List, Tuple

from data_model.asset import AssetSnapshot
from strategy.interfaces import (
    DuplicateScorerError,
    IScorer,
    PoolScore,
    RankingTable,
    ScoringContext,
    ScoringParams,
    WeightConfig,
)

logger = logging.getLogger(__name__)


class ScoringEngine:
    """聚合多 Scorer 输出，加权排序。"""

    def __init__(
        self,
        params: ScoringParams,
        weight_cfg: WeightConfig,
        scorers: List[IScorer] | None = None,
    ) -> None:
        self.params = params
        self._weight_cfg_raw = weight_cfg
        self._scorers: Dict[str, IScorer] = {}
        if scorers:
            for s in scorers:
                self.register(s)

    @property
    def weight_cfg(self) -> WeightConfig:
        """对外暴露归一化后的权重。"""
        return self._weight_cfg_raw.normalized()

    @property
    def scorers(self) -> Tuple[IScorer, ...]:
        # 按 name 字母序返回，便于日志/复现一致
        return tuple(self._scorers[k] for k in sorted(self._scorers))

    # ----- 注册 -----

    def register(self, scorer: IScorer) -> None:
        """注册一个 Scorer。重复注册（同 name）抛 DuplicateScorerError（E-RT-003）。

        幂等说明：调用方若需「替换」应先 unregister 再 register；不在此处静默替换，
        以避免配置变更被掩盖。
        """
        if scorer.name in self._scorers:
            raise DuplicateScorerError(
                f"Scorer name='{scorer.name}' 已注册，禁止重复注册"
            )
        self._scorers[scorer.name] = scorer
        logger.info("registered scorer: %s", scorer.name)

    def unregister(self, name: str) -> None:
        self._scorers.pop(name, None)

    # ----- 主流程 -----

    def run(
        self,
        snapshot: AssetSnapshot,
        ctx: ScoringContext | None = None,
    ) -> RankingTable:
        """对 snapshot 中的所有池打分并排序。

        步骤：
          1. 对每个 Scorer 取其 ScoreVector
          2. 加权求和得到每池综合分（仅注册了的 Scorer 参与；权重缺失视为 0）
          3. 排除 ctx.excluded_pools 中的池
          4. 按 (score desc, pool_id asc) 稳定排序输出 RankingTable
        """
        if not self._scorers:
            raise RuntimeError("ScoringEngine 未注册任何 Scorer，无法 run()")

        ctx = ctx or ScoringContext()
        excluded = set(ctx.excluded_pools)

        # 1. 收集每个 Scorer 的输出
        per_scorer_scores: Dict[str, Dict[str, Decimal]] = {}
        for name, scorer in self._scorers.items():
            sv = scorer.score(snapshot, self.params)
            per_scorer_scores[name] = dict(sv.scores)

        # 2. 加权聚合
        normalized = self.weight_cfg.weights
        candidates = [pid for pid in snapshot.pool_ids() if pid not in excluded]

        ranked: List[PoolScore] = []
        for pid in candidates:
            agg = Decimal(0)
            components: Dict[str, Decimal] = {}
            for name in self._scorers:
                w = normalized.get(name, Decimal(0))
                comp = per_scorer_scores[name].get(pid, Decimal(0))
                components[name] = comp
                agg += w * comp
            ranked.append(PoolScore(pool_id=pid, score=agg, components=components))

        # 3. 稳定排序：按 score 降序，pool_id 升序
        ranked.sort(key=lambda ps: (-ps.score, ps.pool_id))

        return RankingTable(
            snapshot_tick=snapshot.tick,
            rankings=tuple(ranked),
        )

    def top_n(
        self,
        snapshot: AssetSnapshot,
        n: int,
        ctx: ScoringContext | None = None,
    ) -> Tuple[PoolScore, ...]:
        """便捷方法：直接取前 N 名。"""
        return self.run(snapshot, ctx).top_n(n)
