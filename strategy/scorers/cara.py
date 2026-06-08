"""CARA（常绝对风险厌恶）效用评分器。

效用函数：U(r) = -exp(-α · r)，α > 0
  - α 越大 → 越厌恶风险 → 高波动池被相对低估
  - 用当前 APY 作为 r 的代理（小尺度下 -exp(-αr) ≈ αr - α²·r²/2 + ...）

实现细节：
  - 用 numpy.exp 计算（输入 float），最终结果 z-score 归一化为跨池可比
  - 输出仍是 ScoreVector，作为 ScoringEngine 的第 4 个分量
"""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Dict

from data_model.asset import AssetSnapshot
from strategy.interfaces import IScorer, ScoringParams, ScoreVector
from strategy.scorers._common import to_decimal, zscore_dict


def cara_utility(r: float, alpha: float) -> float:
    """U(r) = -exp(-α·r)。α 必须为正。"""
    if alpha <= 0:
        raise ValueError(f"cara_alpha 必须为正，got {alpha}")
    # 防止 -α·r 过大溢出：对极端值 clip
    x = -alpha * r
    if x > 700:   # math.exp(700) 接近 double 上限
        x = 700.0
    if x < -700:
        x = -700.0
    return -math.exp(x)


class CARAUtilityAdjuster(IScorer):
    """CARA 效用评分器。

    输入：每个池的当前 APY（apy_series 末项）
    输出：归一化的效用 z-score。高 APY 池 → 高 U → 高分；α 越大对差异越敏感。
    """

    @property
    def name(self) -> str:
        return "cara"

    def score(self, snapshot: AssetSnapshot, params: ScoringParams) -> ScoreVector:
        alpha_f = float(params.cara_alpha)
        raw: Dict[str, Decimal] = {}

        for pool_id in snapshot.pool_ids():
            pm = snapshot.pools[pool_id]
            current_apy = float(pm.apy_series[-1]) if pm.apy_series else 0.0
            raw[pool_id] = to_decimal(cara_utility(current_apy, alpha_f))

        z = zscore_dict(raw)
        return ScoreVector(scorer_name=self.name, scores=z)
