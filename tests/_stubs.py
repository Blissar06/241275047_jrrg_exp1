"""测试用的 IFrictionEstimator stub。

Phase 3 在 backtest/cost_model.py 提供真实实现；此处是一个可控的常数估算器，
便于 RotationEngine / ReinvestEngine 单测。
"""
from __future__ import annotations

from decimal import Decimal

from data_model.asset import AssetSnapshot
from strategy.interfaces import (
    FrictionBreakdown,
    IFrictionEstimator,
    OperationType,
)


class StubFrictionEstimator(IFrictionEstimator):
    """每次调用返回固定 (gas, slippage, lvr)；可注入异常以测试错误路径。"""

    def __init__(
        self,
        gas: Decimal = Decimal("10"),
        slippage_rate: Decimal = Decimal("0.001"),
        lvr_rate: Decimal = Decimal("0"),
        raise_on_call: bool = False,
        return_negative: bool = False,
    ) -> None:
        self.gas = gas
        self.slippage_rate = slippage_rate
        self.lvr_rate = lvr_rate
        self.raise_on_call = raise_on_call
        self.return_negative = return_negative
        self.call_count = 0

    def estimate(
        self,
        op_type: OperationType,
        amount: Decimal,
        pool_id: str,
        snapshot: AssetSnapshot,
    ) -> FrictionBreakdown:
        self.call_count += 1
        if self.raise_on_call:
            raise RuntimeError("stub configured to raise")
        if self.return_negative:
            return FrictionBreakdown(
                gas=Decimal("-1"), slippage=Decimal(0), lvr=Decimal(0),
            )
        # 复投操作不计入滑点/LVR
        if op_type in (OperationType.REINVEST, OperationType.CLAIM):
            return FrictionBreakdown(gas=self.gas, slippage=Decimal(0), lvr=Decimal(0))
        slip = amount * self.slippage_rate
        lvr = amount * self.lvr_rate
        return FrictionBreakdown(gas=self.gas, slippage=slip, lvr=lvr)
