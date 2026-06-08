"""真实摩擦成本估算器（替代 Phase 2 的 StubFrictionEstimator）。

三分量：
  1. Gas       —— (base_fee + priority_fee) × gas_limit_by_op
                 单位约定：env.gas_*_fee 已折算为「计价本位 / gas-unit」，
                 因此乘以 gas_limit 直接得到计价本位成本。
  2. Slippage  —— 阶梯函数：trade_size / tvl 决定档位
                  - ratio < threshold_low (0.01)              → low_rate
                  - threshold_low ≤ ratio < threshold_high   → mid_rate
                  - ratio ≥ threshold_high (0.05)             → high_rate
                  滑点成本 = trade_size × rate
  3. LVR       —— |oracle_price - pool_price| / oracle_price × trade_size × 0.5

异常处理（E-RT-004）：
  - 内部对每个 (op_type, pool_id) 维护上一 tick 的最近合法 FrictionBreakdown
  - 任一分量为负或抛异常时，回退该缓存；无缓存退回到全 0（保守，不阻断 ROTATE）
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, Optional, Tuple

from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from strategy.interfaces import (
    DataIntegrityError,
    FrictionBreakdown,
    IFrictionEstimator,
    OperationType,
)

logger = logging.getLogger(__name__)


_DEFAULT_GAS_LIMITS: Dict[OperationType, int] = {
    OperationType.ROTATE: 350_000,
    OperationType.REINVEST: 180_000,
    OperationType.CLAIM: 80_000,
    OperationType.DEPOSIT: 200_000,
}


class FrictionEstimator(IFrictionEstimator):
    """三分量摩擦估算器。"""

    def __init__(
        self,
        gas_limits: Optional[Dict[OperationType, int]] = None,
        slip_threshold_low: Decimal = Decimal("0.01"),
        slip_threshold_high: Decimal = Decimal("0.05"),
        slip_rate_low: Decimal = Decimal("0.001"),
        slip_rate_mid: Decimal = Decimal("0.003"),
        slip_rate_high: Decimal = Decimal("0.008"),
    ) -> None:
        self.gas_limits = dict(_DEFAULT_GAS_LIMITS)
        if gas_limits:
            self.gas_limits.update(gas_limits)

        if not (Decimal(0) <= slip_threshold_low <= slip_threshold_high):
            raise ValueError("slippage 阈值必须满足 0 ≤ low ≤ high")
        if min(slip_rate_low, slip_rate_mid, slip_rate_high) < 0:
            raise ValueError("slippage 各档费率必须 ≥ 0")

        self.slip_threshold_low = slip_threshold_low
        self.slip_threshold_high = slip_threshold_high
        self.slip_rate_low = slip_rate_low
        self.slip_rate_mid = slip_rate_mid
        self.slip_rate_high = slip_rate_high

        self._cache: Dict[Tuple[OperationType, str], FrictionBreakdown] = {}

    # ---------- 工厂 ----------

    @classmethod
    def from_config(cls, config: dict) -> "FrictionEstimator":
        """从 config.yaml 风格的 dict 构建实例。"""
        gas_limits_cfg = config.get("gas_limits", {}) or {}
        gas_limits = {}
        for k, v in gas_limits_cfg.items():
            try:
                op = OperationType[k]
            except KeyError:
                logger.warning("config.gas_limits 含未知操作 %s，已忽略", k)
                continue
            gas_limits[op] = int(v)

        steps = config.get("slippage_steps", {}) or {}
        return cls(
            gas_limits=gas_limits,
            slip_rate_low=Decimal(str(steps.get("low", "0.001"))),
            slip_rate_mid=Decimal(str(steps.get("mid", "0.003"))),
            slip_rate_high=Decimal(str(steps.get("high", "0.008"))),
        )

    # ---------- 三分量估算（独立暴露，便于报表层细粒度归因） ----------

    def estimate_gas(self, op_type: OperationType, env: EnvSnapshot) -> Decimal:
        gas_limit = self.gas_limits.get(op_type)
        if gas_limit is None:
            logger.warning("未知 OperationType=%s，使用默认 100000", op_type)
            gas_limit = 100_000
        return (env.gas_base_fee + env.gas_priority_fee) * Decimal(gas_limit)

    def estimate_slippage(self, trade_size: Decimal, tvl: Decimal) -> Decimal:
        if trade_size <= 0:
            return Decimal(0)
        if tvl <= 0:
            # 无流动性：直接用最高档（保守惩罚）
            return trade_size * self.slip_rate_high
        ratio = trade_size / tvl
        if ratio < self.slip_threshold_low:
            rate = self.slip_rate_low
        elif ratio < self.slip_threshold_high:
            rate = self.slip_rate_mid
        else:
            rate = self.slip_rate_high
        return trade_size * rate

    def estimate_lvr(
        self,
        oracle_price: Decimal,
        pool_price: Decimal,
        trade_size: Decimal,
    ) -> Decimal:
        if oracle_price <= 0 or trade_size <= 0:
            return Decimal(0)
        diff = abs(oracle_price - pool_price)
        return diff / oracle_price * trade_size * Decimal("0.5")

    # ---------- 接口实现 ----------

    def estimate(
        self,
        op_type: OperationType,
        amount: Decimal,
        pool_id: str,
        snapshot: AssetSnapshot,
    ) -> FrictionBreakdown:
        cache_key = (op_type, pool_id)
        try:
            pool: Optional[PoolMetrics] = snapshot.pools.get(pool_id)
            if pool is None:
                raise DataIntegrityError(
                    f"snapshot.pools 缺少 pool_id={pool_id}"
                )

            gas = self.estimate_gas(op_type, snapshot.env)

            # 复投 / 领取奖励：链上不发生大额交易，仅 gas
            if op_type in (OperationType.REINVEST, OperationType.CLAIM):
                fb = FrictionBreakdown(gas=gas, slippage=Decimal(0), lvr=Decimal(0))
            else:
                slip = self.estimate_slippage(amount, pool.tvl)
                oracle = snapshot.env.oracle_price.get(pool_id, pool.token_price)
                lvr = self.estimate_lvr(oracle, pool.token_price, amount)
                fb = FrictionBreakdown(gas=gas, slippage=slip, lvr=lvr)

            # 任一分量为负 → 回退缓存
            if fb.gas < 0 or fb.slippage < 0 or fb.lvr < 0:
                cached = self._cache.get(cache_key)
                logger.warning(
                    "tick=%d friction 含负值 %s, 回退缓存=%s",
                    snapshot.env.tick, fb, cached,
                )
                return cached or FrictionBreakdown(Decimal(0), Decimal(0), Decimal(0))

            self._cache[cache_key] = fb
            return fb

        except Exception as e:
            cached = self._cache.get(cache_key)
            logger.warning(
                "tick=%d FrictionEstimator 异常 op=%s pool=%s: %s; 回退缓存",
                snapshot.env.tick, op_type.value, pool_id, e,
            )
            return cached or FrictionBreakdown(Decimal(0), Decimal(0), Decimal(0))
