"""压力事件注入（FR-07）。

支持三类事件：
  - GAS_SPIKE       env.gas_base_fee × (1 + impact_ratio)
  - POOL_EXPLOIT    target 池 apy_series 末项 × (1 - impact_ratio)，tvl × (1 - impact_ratio)
  - LIQUIDITY_DRYUP target 池 tvl × (1 - impact_ratio)（apy 不变）

apply() 返回新 AssetSnapshot —— 不修改原对象（NFR-02）。

事件计划支持从 YAML / JSON 文件加载：

    events:
      - event_type: GAS_SPIKE
        start_tick: 150
        duration: 5
        impact_ratio: "4.0"
      - event_type: POOL_EXPLOIT
        start_tick: 200
        duration: 1
        target_pool_id: pool_a
        impact_ratio: "0.9"
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics

logger = logging.getLogger(__name__)


class EventType(Enum):
    GAS_SPIKE = "GAS_SPIKE"
    POOL_EXPLOIT = "POOL_EXPLOIT"
    LIQUIDITY_DRYUP = "LIQUIDITY_DRYUP"


@dataclass(frozen=True, slots=True)
class StressEvent:
    """单个压力事件描述。所有字段不可变。"""

    event_type: EventType
    start_tick: int
    duration: int
    impact_ratio: Decimal
    target_pool_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"duration 必须为正，got {self.duration}")
        if self.start_tick < 0:
            raise ValueError(f"start_tick 必须非负，got {self.start_tick}")
        if self.event_type in (EventType.POOL_EXPLOIT, EventType.LIQUIDITY_DRYUP) \
                and self.target_pool_id is None:
            raise ValueError(
                f"{self.event_type.value} 必须指定 target_pool_id"
            )

    def is_active(self, tick: int) -> bool:
        return self.start_tick <= tick < self.start_tick + self.duration


class EventInjector:
    """根据事件计划在快照上应用压力影响。"""

    def __init__(self, schedule: Optional[List[StressEvent]] = None) -> None:
        self._schedule: List[StressEvent] = list(schedule) if schedule else []

    @property
    def schedule(self) -> List[StressEvent]:
        return list(self._schedule)

    def add(self, event: StressEvent) -> None:
        self._schedule.append(event)

    def active_at(self, tick: int) -> List[StressEvent]:
        return [e for e in self._schedule if e.is_active(tick)]

    # ---------- 加载 ----------

    @classmethod
    def load_schedule(cls, path: str | Path) -> "EventInjector":
        """从 YAML / JSON 文件加载事件计划。"""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif p.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            raise ValueError(f"不支持的事件计划文件格式：{p.suffix}")

        if not isinstance(data, dict) or "events" not in data:
            raise ValueError("事件计划必须包含顶层 'events' 列表")

        events: List[StressEvent] = []
        for raw in data["events"]:
            try:
                ev = StressEvent(
                    event_type=EventType[raw["event_type"]],
                    start_tick=int(raw["start_tick"]),
                    duration=int(raw["duration"]),
                    impact_ratio=Decimal(str(raw["impact_ratio"])),
                    target_pool_id=raw.get("target_pool_id"),
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"事件计划条目非法：{raw}: {e}")
            events.append(ev)

        logger.info("loaded %d stress events from %s", len(events), p)
        return cls(events)

    # ---------- 应用 ----------

    def apply(self, snapshot: AssetSnapshot) -> AssetSnapshot:
        """对 snapshot 应用当前 tick 上所有 active 事件，返回新对象。"""
        active = self.active_at(snapshot.tick)
        if not active:
            return snapshot

        new_pools: Dict[str, PoolMetrics] = dict(snapshot.pools)
        new_gas_base = snapshot.env.gas_base_fee
        new_gas_prio = snapshot.env.gas_priority_fee

        for ev in active:
            if ev.event_type == EventType.GAS_SPIKE:
                new_gas_base = new_gas_base * (Decimal(1) + ev.impact_ratio)
                new_gas_prio = new_gas_prio * (Decimal(1) + ev.impact_ratio)

            elif ev.event_type == EventType.POOL_EXPLOIT:
                pool = new_pools.get(ev.target_pool_id)
                if pool is None:
                    logger.warning(
                        "tick=%d POOL_EXPLOIT target_pool=%s 不在 snapshot 中，跳过",
                        snapshot.tick, ev.target_pool_id,
                    )
                    continue
                new_pools[ev.target_pool_id] = _hit_pool(
                    pool,
                    apy_factor=Decimal(1) - ev.impact_ratio,
                    tvl_factor=Decimal(1) - ev.impact_ratio,
                    new_gas_base=new_gas_base,
                )

            elif ev.event_type == EventType.LIQUIDITY_DRYUP:
                pool = new_pools.get(ev.target_pool_id)
                if pool is None:
                    logger.warning(
                        "tick=%d LIQUIDITY_DRYUP target_pool=%s 不存在，跳过",
                        snapshot.tick, ev.target_pool_id,
                    )
                    continue
                new_pools[ev.target_pool_id] = _hit_pool(
                    pool,
                    apy_factor=Decimal(1),  # APY 保持
                    tvl_factor=Decimal(1) - ev.impact_ratio,
                    new_gas_base=new_gas_base,
                )

        # 重建 env：oracle_price / timestamp / tick 不变
        new_env = EnvSnapshot(
            tick=snapshot.env.tick,
            timestamp=snapshot.env.timestamp,
            oracle_price=dict(snapshot.env.oracle_price),
            gas_base_fee=new_gas_base,
            gas_priority_fee=new_gas_prio,
        )

        # 池上的 gas_base_fee 字段也要同步更新（避免与 env 不一致）
        if new_gas_base != snapshot.env.gas_base_fee:
            for pid, pm in list(new_pools.items()):
                if pm.gas_base_fee != new_gas_base:
                    new_pools[pid] = PoolMetrics(
                        pool_id=pm.pool_id,
                        apy_series=pm.apy_series,
                        tvl=pm.tvl,
                        vol_30d=pm.vol_30d,
                        token_price=pm.token_price,
                        gas_base_fee=new_gas_base,
                    )

        return AssetSnapshot(tick=snapshot.tick, pools=new_pools, env=new_env)


def _hit_pool(
    pm: PoolMetrics,
    apy_factor: Decimal,
    tvl_factor: Decimal,
    new_gas_base: Decimal,
) -> PoolMetrics:
    """构造一个被压力事件冲击后的 PoolMetrics。

    apy_factor 仅作用于 apy_series 末项（当前 tick），历史不修改。
    """
    if pm.apy_series:
        new_apy_series = pm.apy_series[:-1] + (pm.apy_series[-1] * apy_factor,)
    else:
        new_apy_series = pm.apy_series

    return PoolMetrics(
        pool_id=pm.pool_id,
        apy_series=new_apy_series,
        tvl=pm.tvl * tvl_factor,
        vol_30d=pm.vol_30d,
        token_price=pm.token_price,
        gas_base_fee=new_gas_base,
    )
