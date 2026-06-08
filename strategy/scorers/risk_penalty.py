"""风险惩罚评分器：下行波动 + 最大回撤。

按 spec 拆为 2 个独立 Scorer，对应 WeightConfig 的 vol_penalty / mdd_penalty 两个 key：
  - DownsideVolPenaltyScorer：apy diff 的负值部分标准差
  - MaxDrawdownPenaltyScorer：apy_series 的最大回撤

两者均「先算 raw 风险量 → 跨池 z-score → 取负」，使得风险大的池得低分。
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict

import numpy as np

from data_model.asset import AssetSnapshot
from strategy.interfaces import IScorer, ScoringParams, ScoreVector
from strategy.scorers._common import to_decimal, zscore_dict

logger = logging.getLogger(__name__)


def downside_volatility(apy_series_floats: np.ndarray) -> float:
    """下行波动率：仅对 apy diff < 0 的项做总体标准差。

    序列长度 < 2 时无 diff，返回 0；无负 diff 也返回 0。
    """
    if apy_series_floats.size < 2:
        return 0.0
    diffs = np.diff(apy_series_floats)
    neg = diffs[diffs < 0]
    if neg.size == 0:
        return 0.0
    return float(np.std(neg, ddof=0))


def max_drawdown(apy_series_floats: np.ndarray) -> float:
    """对 apy_series 计算 1 - x / cummax(x) 的最大值。

    序列长度 < 1 或 cummax<=0 时返回 0；APY 视为「值序列」，
    回撤反映该池历史 APY 从峰值的相对跌幅。
    """
    if apy_series_floats.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(apy_series_floats)
    # 防 0 除：peak<=0 时该位置回撤设为 0
    safe_peaks = np.where(peaks > 0, peaks, 1.0)
    drawdowns = np.where(peaks > 0, 1.0 - apy_series_floats / safe_peaks, 0.0)
    return float(np.max(drawdowns))


class DownsideVolPenaltyScorer(IScorer):
    """下行波动率惩罚：raw 越大 → z-score 取负后越低。"""

    @property
    def name(self) -> str:
        return "vol_penalty"

    def score(self, snapshot: AssetSnapshot, params: ScoringParams) -> ScoreVector:
        raw: Dict[str, Decimal] = {}
        win = params.vol_window
        for pool_id in snapshot.pool_ids():
            pm = snapshot.pools[pool_id]
            window_slice = pm.apy_series[-win:] if win > 0 else pm.apy_series
            arr = np.array([float(x) for x in window_slice], dtype=np.float64)
            dv = downside_volatility(arr)
            if dv == 0.0 and arr.size >= 2:
                # 无下行风险并不是异常，但日志便于排查
                logger.debug("pool=%s downside_vol=0", pool_id)
            raw[pool_id] = to_decimal(dv)

        z = zscore_dict(raw)
        # 取负：风险高 → 得分低
        return ScoreVector(
            scorer_name=self.name,
            scores={k: -v for k, v in z.items()},
        )


class MaxDrawdownPenaltyScorer(IScorer):
    """最大回撤惩罚：raw 越大 → z-score 取负后越低。"""

    @property
    def name(self) -> str:
        return "mdd_penalty"

    def score(self, snapshot: AssetSnapshot, params: ScoringParams) -> ScoreVector:
        raw: Dict[str, Decimal] = {}
        win = params.mdd_window
        for pool_id in snapshot.pool_ids():
            pm = snapshot.pools[pool_id]
            window_slice = pm.apy_series[-win:] if win > 0 else pm.apy_series
            arr = np.array([float(x) for x in window_slice], dtype=np.float64)
            raw[pool_id] = to_decimal(max_drawdown(arr))

        z = zscore_dict(raw)
        return ScoreVector(
            scorer_name=self.name,
            scores={k: -v for k, v in z.items()},
        )


class TokenPriceVolPenaltyScorer(IScorer):
    """Token 价格下行波动惩罚 —— 捕捉本金贬值风险。

    与 DownsideVolPenaltyScorer 的区别：那个看 APY 序列的波动（收益率风险），
    这个看 token_price 序列的下行波动（本金价格风险）。
    PoolMetrics.token_price_series 为空时，该池 raw 设为 0，由 z-score 自然吸收。
    """

    @property
    def name(self) -> str:
        return "price_vol_penalty"

    def score(self, snapshot: AssetSnapshot, params: ScoringParams) -> ScoreVector:
        raw: Dict[str, Decimal] = {}
        win = params.vol_window
        for pool_id in snapshot.pool_ids():
            pm = snapshot.pools[pool_id]
            series = pm.token_price_series
            if not series:
                raw[pool_id] = Decimal(0)
                continue
            window_slice = series[-win:] if win > 0 else series
            # token_price 的对数收益率作为「价格变化率」
            arr = np.array([float(x) for x in window_slice], dtype=np.float64)
            if arr.size < 2 or (arr <= 0).any():
                raw[pool_id] = Decimal(0)
                continue
            log_rets = np.diff(np.log(arr))
            neg = log_rets[log_rets < 0]
            if neg.size == 0:
                raw[pool_id] = Decimal(0)
            else:
                raw[pool_id] = to_decimal(float(np.std(neg, ddof=0)))

        z = zscore_dict(raw)
        return ScoreVector(
            scorer_name=self.name,
            scores={k: -v for k, v in z.items()},
        )


class TokenPriceMDDPenaltyScorer(IScorer):
    """Token 价格最大回撤惩罚 —— 捕捉本金贬值风险。"""

    @property
    def name(self) -> str:
        return "price_mdd_penalty"

    def score(self, snapshot: AssetSnapshot, params: ScoringParams) -> ScoreVector:
        raw: Dict[str, Decimal] = {}
        win = params.mdd_window
        for pool_id in snapshot.pool_ids():
            pm = snapshot.pools[pool_id]
            series = pm.token_price_series
            if not series:
                raw[pool_id] = Decimal(0)
                continue
            window_slice = series[-win:] if win > 0 else series
            arr = np.array([float(x) for x in window_slice], dtype=np.float64)
            raw[pool_id] = to_decimal(max_drawdown(arr))

        z = zscore_dict(raw)
        return ScoreVector(
            scorer_name=self.name,
            scores={k: -v for k, v in z.items()},
        )
