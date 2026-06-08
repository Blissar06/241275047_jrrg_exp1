"""不可变值对象：PoolMetrics / EnvSnapshot / AssetSnapshot

所有金融计算字段使用 Decimal，不使用 float — 满足 NFR-01（精度 28 位）。
对外暴露的容器字段统一包装为 MappingProxyType / tuple — 满足 NFR-02（结果绝对复现）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Iterable, Mapping, Tuple


@dataclass(frozen=True, slots=True)
class PoolMetrics:
    """单个流动性池在某一 tick 的快照指标。"""

    pool_id: str
    apy_series: Tuple[Decimal, ...]   # 截至当前 tick 的回看 APY 序列（末项为当前）
    tvl: Decimal                      # 当前池锁仓量
    vol_30d: Decimal                  # 30 日历史波动率
    token_price: Decimal              # 代币价格（计价本位）
    gas_base_fee: Decimal             # 当前 tick 的 base fee（冗余字段，便于 per-pool 决策）
    # 截至当前 tick 的回看 token_price 序列（末项 = token_price）；空 tuple 表示
    # 未提供（向后兼容旧测试与不携带价格历史的 loader）。Scorer 在序列为空时
    # 应当退化为 0 分而不抛错。
    token_price_series: Tuple[Decimal, ...] = ()

    def __post_init__(self) -> None:
        # 强制 apy_series / token_price_series 为 tuple
        if not isinstance(self.apy_series, tuple):
            object.__setattr__(self, "apy_series", tuple(self.apy_series))
        if not isinstance(self.token_price_series, tuple):
            object.__setattr__(self, "token_price_series", tuple(self.token_price_series))
        # 类型校验仅做最低限度检查 —— 上层 loader 已经规范化。
        for name in ("tvl", "vol_30d", "token_price", "gas_base_fee"):
            v = getattr(self, name)
            if not isinstance(v, Decimal):
                raise TypeError(f"PoolMetrics.{name} must be Decimal, got {type(v).__name__}")


@dataclass(frozen=True, slots=True)
class EnvSnapshot:
    """环境状态：时间、oracle 价格、Gas 行情。"""

    tick: int
    timestamp: datetime
    oracle_price: Mapping[str, Decimal]   # symbol -> price（计价本位）
    gas_base_fee: Decimal
    gas_priority_fee: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.oracle_price, MappingProxyType):
            object.__setattr__(
                self,
                "oracle_price",
                MappingProxyType(dict(self.oracle_price)),
            )
        for name in ("gas_base_fee", "gas_priority_fee"):
            v = getattr(self, name)
            if not isinstance(v, Decimal):
                raise TypeError(f"EnvSnapshot.{name} must be Decimal, got {type(v).__name__}")


@dataclass(frozen=True, slots=True)
class AssetSnapshot:
    """单个时间步上所有池 + 环境的复合快照（回测引擎的最小输入单元）。"""

    tick: int
    pools: Mapping[str, PoolMetrics]
    env: EnvSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.pools, MappingProxyType):
            object.__setattr__(
                self,
                "pools",
                MappingProxyType(dict(self.pools)),
            )

    def pool_ids(self) -> Tuple[str, ...]:
        """返回所有池 id（确定性顺序，便于复现）。"""
        return tuple(sorted(self.pools.keys()))


def freeze_decimal_iter(values: Iterable) -> Tuple[Decimal, ...]:
    """工具函数：把任意数值可迭代对象规范化为 Tuple[Decimal, ...]。"""
    return tuple(v if isinstance(v, Decimal) else Decimal(str(v)) for v in values)
