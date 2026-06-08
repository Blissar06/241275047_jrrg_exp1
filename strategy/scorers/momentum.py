"""动量评分器：EWMA(apy_series) → z-score。

公式：M_t = λ·M_{t-1} + (1-λ)·r_t，r_t = apy_series[t]
λ ∈ (0, 1)；λ 越大越平滑、越偏向历史。
"""
from __future__ import annotations

from decimal import Decimal
from typing import Dict, Tuple

from data_model.asset import AssetSnapshot
from strategy.interfaces import IScorer, ScoringParams, ScoreVector
from strategy.scorers._common import zscore_dict


def ewma(series: Tuple[Decimal, ...], lam: Decimal) -> Decimal:
    """对 Decimal 序列做 EWMA。空序列返回 0；单元素直接返回该值。"""
    if not series:
        return Decimal(0)
    if not (Decimal(0) < lam < Decimal(1)):
        raise ValueError(f"momentum_lambda 必须在 (0,1)，got {lam}")
    m = series[0]
    one_minus = Decimal(1) - lam
    for r in series[1:]:
        m = lam * m + one_minus * r
    return m


class MomentumScorer(IScorer):
    """对每个池的 apy_series 计算 EWMA，跨池 z-score 归一化。"""

    @property
    def name(self) -> str:
        return "momentum"

    def score(self, snapshot: AssetSnapshot, params: ScoringParams) -> ScoreVector:
        raw: Dict[str, Decimal] = {}
        win = params.momentum_window
        for pool_id in snapshot.pool_ids():
            pm = snapshot.pools[pool_id]
            # 仅取最近 momentum_window 个观测，多于则截断
            window_slice = pm.apy_series[-win:] if win > 0 else pm.apy_series
            raw[pool_id] = ewma(window_slice, params.momentum_lambda)

        scores = zscore_dict(raw)
        return ScoreVector(scorer_name=self.name, scores=scores)
