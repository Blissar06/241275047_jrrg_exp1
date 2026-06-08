"""预定义策略原型（StrategyPreset）。

设计思路 —— 基于详细设计文档定义的算法原语（Momentum / RiskPenalty / CARA /
Threshold Rotation）覆盖典型的「策略风格象限」，让用户能一键载入有意义的
参数组合，而不必从零调参。

策略象限：

       ▲  高频
       │   ┌──────────────┐
       │   │ 激进动量      │
       │   │  τ↓ thr↓ 动量↑ │
       │   └──────────────┘
       │   ┌──────────────┐
       │   │   均衡        │ (default)
       │   │  各因子均匀   │
       │   └──────────────┘
       │   ┌──────────────┐  ┌──────────────┐
       │   │ 保守稳健      │  │ 极端风险厌恶  │
       │   │  τ↑ thr↑ MDD↑ │  │  α↑↑ VOL↑↑   │
       │   └──────────────┘  └──────────────┘
       │   ┌──────────────┐
       │   │ 低频价值      │
       │   │  τ↑↑ thr↑↑    │
       │   └──────────────┘
       ▼  低频
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List


@dataclass(frozen=True, slots=True)
class StrategyPreset:
    """命名策略预设。"""

    name: str
    description: str
    # 评分权重（自动归一化，但提供时应已大致平衡）
    weights: Dict[str, Decimal]
    # 轮动门槛
    threshold: Decimal              # 调仓增益相对本金比例
    tau_reset: Decimal              # τ-reset 偏离度
    cara_alpha: Decimal             # CARA 风险厌恶
    # 增益估算的前瞻窗口（tick）；保守策略用更长窗口让 gain 有机会跨过 threshold
    gain_horizon_ticks: int = 30
    # 滑点档位 key（对应 ui.app.SLIPPAGE_PRESETS 的键）
    slip_choice: str = "中（默认）"
    # 适用提示，UI 中展示
    hint: str = ""


def _w(
    mom: float, vol: float, mdd: float, cara: float,
    price_vol: float = 0.0, price_mdd: float = 0.0,
) -> Dict[str, Decimal]:
    return {
        "momentum": Decimal(str(mom)),
        "vol_penalty": Decimal(str(vol)),
        "mdd_penalty": Decimal(str(mdd)),
        "cara": Decimal(str(cara)),
        "price_vol_penalty": Decimal(str(price_vol)),
        "price_mdd_penalty": Decimal(str(price_mdd)),
    }


# =================================================================
# 5 个内置预设
# =================================================================

CONSERVATIVE = StrategyPreset(
    name="保守稳健",
    description=(
        "高 τ + 中等 threshold + 重 MDD 惩罚 + 长前瞻窗口。"
        "回撤敏感、调仓罕见，适合追求资本保全的场景。"
    ),
    weights=_w(
        mom=0.05, vol=0.15, mdd=0.15, cara=0.10,
        price_vol=0.25, price_mdd=0.30,         # 重价格风险
    ),
    threshold=Decimal("0.002"),      # 0.2% 本金净增益门槛
    tau_reset=Decimal("0.10"),       # 要 +10% APY 才动
    cara_alpha=Decimal("5.0"),       # 高风险厌恶
    gain_horizon_ticks=90,            # 用 90 天前瞻让 gain 有机会跨阈值
    slip_choice="中（默认）",
    hint="目标 MDD < 5%、夏普 > 1.0；预期调仓 < 5 次。",
)

BALANCED = StrategyPreset(
    name="均衡（默认）",
    description=(
        "文档默认配置。各因子均匀贡献，轮动门槛适中。"
        "在大多数市场状态下应能稳定运行。"
    ),
    weights=_w(
        mom=0.30, vol=0.15, mdd=0.15, cara=0.10,
        price_vol=0.15, price_mdd=0.15,
    ),
    threshold=Decimal("0.001"),
    tau_reset=Decimal("0.05"),
    cara_alpha=Decimal("2.0"),
    slip_choice="中（默认）",
    hint="所有指标均衡；调参基准；预期年化与摩擦平衡。",
)

AGGRESSIVE_MOMENTUM = StrategyPreset(
    name="激进动量",
    description=(
        "压低 τ 与 threshold，让策略尽可能跟随 APY 顶部；动量权重远高于其他因子。"
        "在趋势明显的行情上理论收益最高，但摩擦成本与回撤都偏大。"
    ),
    weights=_w(
        mom=0.60, vol=0.10, mdd=0.10, cara=0.10,
        price_vol=0.05, price_mdd=0.05,         # 轻价格风险
    ),
    threshold=Decimal("0.0005"),
    tau_reset=Decimal("0.02"),
    cara_alpha=Decimal("1.0"),
    slip_choice="低（流动性充足）",
    hint="预期调仓次数高（> 30）；摩擦占比 > 5%；MDD 可能较大。",
)

LOW_FREQUENCY_VALUE = StrategyPreset(
    name="低频价值",
    description=(
        "高 τ 与 threshold，只有出现显著机会才换仓。"
        "搭配长前瞻窗口；适合 Gas 高企或调仓成本不容忽视的场景。"
    ),
    weights=_w(
        mom=0.20, vol=0.20, mdd=0.20, cara=0.10,
        price_vol=0.15, price_mdd=0.15,
    ),
    threshold=Decimal("0.005"),      # 0.5% 本金净增益门槛
    tau_reset=Decimal("0.15"),       # 要 +15% APY 才动
    cara_alpha=Decimal("2.0"),
    gain_horizon_ticks=120,           # 用 120 天窗口看 gain
    slip_choice="高（流动性紧张）",
    hint="预期调仓 < 3 次；摩擦占比 < 1%；可能错过短期机会。",
)

EXTREME_RISK_AVERSE = StrategyPreset(
    name="极端风险厌恶",
    description=(
        "拉满 CARA α 与 vol/MDD 惩罚权重，几乎只看下行风险。"
        "适合压力测试与极端避险偏好的研究场景。"
    ),
    weights=_w(
        mom=0.05, vol=0.15, mdd=0.15, cara=0.10,
        price_vol=0.25, price_mdd=0.30,         # 重价格风险
    ),
    threshold=Decimal("0.005"),
    tau_reset=Decimal("0.10"),
    cara_alpha=Decimal("8.0"),
    slip_choice="中（默认）",
    hint="目标 MDD 最低；可能长时间持有低 APY 池。",
)


# =================================================================
# 注册表
# =================================================================

ALL_PRESETS: List[StrategyPreset] = [
    BALANCED,
    CONSERVATIVE,
    AGGRESSIVE_MOMENTUM,
    LOW_FREQUENCY_VALUE,
    EXTREME_RISK_AVERSE,
]

PRESET_REGISTRY: Dict[str, StrategyPreset] = {p.name: p for p in ALL_PRESETS}


def get_preset(name: str) -> StrategyPreset:
    """按名查 preset，未知 name 抛 KeyError。"""
    if name not in PRESET_REGISTRY:
        raise KeyError(
            f"未知预设 {name!r}；可用：{list(PRESET_REGISTRY)}"
        )
    return PRESET_REGISTRY[name]


def list_preset_names() -> List[str]:
    return [p.name for p in ALL_PRESETS]
