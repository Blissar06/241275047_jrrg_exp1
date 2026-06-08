"""默认 IGainEstimator 实现：基于 APY 差和年化系数的简单线性外推。

公式：
  rotation_gain = principal × (target_apy - current_apy) × horizon / ticks_per_year
  reinvest_gain = pending_reward × current_apy × horizon / ticks_per_year

该模型不考虑复利、池容量衰减等高阶效应；这些校正应在 BacktestEngine 中按需追加。
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from data_model.asset import AssetSnapshot
from strategy.interfaces import IGainEstimator, Position


class APYDeltaGainEstimator(IGainEstimator):
    """APY 差 + 可选 token_price 漂移的增益估算。

    V2 改进：当 PoolMetrics 携带足够长的 token_price_series 时，把最近的价格
    漂移按年化外推后并入 effective APY；让正在贬值的池被「正确地避开」、贬值
    的当前持仓被「正确地切出」。

    若 use_price_drift=False 或序列长度 < drift_window，退化为纯 APY 模型。
    """

    def __init__(
        self,
        ticks_per_year: int = 365,
        use_price_drift: bool = True,
        drift_window: int = 14,
    ) -> None:
        if ticks_per_year <= 0:
            raise ValueError(f"ticks_per_year 必须为正，got {ticks_per_year}")
        if drift_window < 2:
            raise ValueError(f"drift_window 必须 ≥ 2，got {drift_window}")
        self.ticks_per_year = ticks_per_year
        self.use_price_drift = use_price_drift
        self.drift_window = drift_window

    # ----- helpers -----

    def _annualized_price_drift(
        self,
        pool_id: str,
        snapshot: AssetSnapshot,
    ) -> Decimal:
        """从 token_price_series 末段 drift_window 个点估年化漂移。

        无序列或序列太短 → 返回 0。
        线性近似：(price_end / price_start - 1) × (ticks_per_year / window_actual)
        """
        if pool_id not in snapshot.pools:
            return Decimal(0)
        series = snapshot.pools[pool_id].token_price_series
        if not series or len(series) < self.drift_window:
            return Decimal(0)
        win = series[-self.drift_window:]
        start = win[0]
        end = win[-1]
        if start <= 0:
            return Decimal(0)
        rel_change = (end - start) / start          # 窗口内相对变化
        # 用窗口实际长度 - 1（差分数）年化
        return rel_change * Decimal(self.ticks_per_year) / Decimal(len(win) - 1)

    def _effective_apy(
        self,
        pool_id: Optional[str],
        snapshot: AssetSnapshot,
    ) -> Decimal:
        if pool_id is None or pool_id not in snapshot.pools:
            return Decimal(0)
        series = snapshot.pools[pool_id].apy_series
        apy = series[-1] if series else Decimal(0)
        if not self.use_price_drift:
            return apy
        drift = self._annualized_price_drift(pool_id, snapshot)
        return apy + drift

    # 旧方法保留为向后兼容（不再使用，但旧测试可能在用）
    def _current_apy(self, position: Position, snapshot: AssetSnapshot) -> Decimal:
        if position.pool_id is None or position.pool_id not in snapshot.pools:
            return Decimal(0)
        series = snapshot.pools[position.pool_id].apy_series
        return series[-1] if series else Decimal(0)

    def _target_apy(self, target_pool_id: str, snapshot: AssetSnapshot) -> Decimal:
        if target_pool_id not in snapshot.pools:
            return Decimal(0)
        series = snapshot.pools[target_pool_id].apy_series
        return series[-1] if series else Decimal(0)

    # ----- 主入口 -----

    def expected_rotation_gain(
        self,
        position: Position,
        target_pool_id: str,
        snapshot: AssetSnapshot,
        horizon_ticks: int,
    ) -> Decimal:
        if horizon_ticks <= 0:
            return Decimal(0)
        invested = position.principal + position.cash + position.pending_reward
        if invested <= 0:
            return Decimal(0)
        target_eff = self._effective_apy(target_pool_id, snapshot)
        current_eff = self._effective_apy(position.pool_id, snapshot)
        delta = target_eff - current_eff
        return invested * delta * Decimal(horizon_ticks) / Decimal(self.ticks_per_year)

    def expected_reinvest_gain(
        self,
        position: Position,
        snapshot: AssetSnapshot,
        horizon_ticks: int,
    ) -> Decimal:
        if horizon_ticks <= 0 or position.pending_reward <= 0:
            return Decimal(0)
        apy = self._current_apy(position, snapshot)
        if apy <= 0:
            return Decimal(0)
        return position.pending_reward * apy * Decimal(horizon_ticks) / Decimal(self.ticks_per_year)
