"""策略层抽象接口与值对象。

设计原则：
  - 所有 Scorer 实现纯函数语义：相同 (snapshot, params) → 相同 ScoreVector
  - 所有值对象 frozen，参与排序的容器为 tuple，保证 NFR-02 可复现
  - 金额/分数最终结果使用 Decimal，中间统计可用 numpy float
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from data_model.asset import AssetSnapshot, EnvSnapshot


# =================================================================
# 值对象
# =================================================================

@dataclass(frozen=True, slots=True)
class ScoringParams:
    """评分参数。来自 config.yaml，运行期不变。"""

    momentum_window: int = 14
    momentum_lambda: Decimal = Decimal("0.85")
    vol_window: int = 30
    mdd_window: int = 90
    cara_alpha: Decimal = Decimal("2.0")


@dataclass(frozen=True, slots=True)
class ScoringContext:
    """单次评分调用的上下文（位置、可用候选等）。"""

    current_position: Optional[str] = None    # 当前持仓 pool_id（可空）
    excluded_pools: Tuple[str, ...] = ()      # 黑名单（受灾池）


@dataclass(frozen=True, slots=True)
class ScoreVector:
    """单个 Scorer 输出：pool_id -> 分数。"""

    scorer_name: str
    scores: Mapping[str, Decimal]

    def __post_init__(self) -> None:
        if not isinstance(self.scores, MappingProxyType):
            object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


@dataclass(frozen=True, slots=True)
class PoolScore:
    """排序条目：pool_id + 综合得分 + 各分量明细。"""

    pool_id: str
    score: Decimal
    components: Mapping[str, Decimal]   # scorer_name -> raw（未加权）分量

    def __post_init__(self) -> None:
        if not isinstance(self.components, MappingProxyType):
            object.__setattr__(
                self,
                "components",
                MappingProxyType(dict(self.components)),
            )


@dataclass(frozen=True, slots=True)
class RankingTable:
    """ScoringEngine.run() 输出。rankings 已按 score 降序、同分按 pool_id 字典序排列。"""

    snapshot_tick: int
    rankings: Tuple[PoolScore, ...]

    def top_n(self, n: int) -> Tuple[PoolScore, ...]:
        if n <= 0:
            return ()
        return self.rankings[:n]

    def get(self, pool_id: str) -> Optional[PoolScore]:
        for ps in self.rankings:
            if ps.pool_id == pool_id:
                return ps
        return None


@dataclass(frozen=True, slots=True)
class WeightConfig:
    """各 Scorer 的聚合权重。"""

    weights: Mapping[str, Decimal]

    def __post_init__(self) -> None:
        if not isinstance(self.weights, MappingProxyType):
            object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))

    def normalized(self) -> "WeightConfig":
        """返回归一化（总和=1）后的新 WeightConfig；总和≤0 抛错。"""
        total = sum(self.weights.values(), Decimal(0))
        if total <= 0:
            raise ValueError(f"WeightConfig 总和必须 > 0，got {total}")
        return WeightConfig({k: v / total for k, v in self.weights.items()})


# =================================================================
# 异常
# =================================================================

class DuplicateScorerError(Exception):
    """重复注册同名 Scorer 时抛出（E-RT-003）。"""


class ScoringError(Exception):
    """评分阶段的可恢复错误（数据缺失等）。"""


class DataIntegrityError(Exception):
    """快照字段缺失或非法（E-RT-001）。引擎应进入 ERROR 状态、fallback=HOLD。"""


# =================================================================
# 仓位与日志（轮动 / 复投共用）
# =================================================================

class OperationType(Enum):
    """链上操作类型。用于 Gas 估算分档与日志归类。"""

    DEPOSIT = "DEPOSIT"      # 首次存入
    ROTATE = "ROTATE"        # 调仓（退出 + 入场）
    REINVEST = "REINVEST"    # 复投
    CLAIM = "CLAIM"          # 仅领取奖励


@dataclass(frozen=True, slots=True)
class Position:
    """当前持仓快照（不可变；状态推进通过返回新对象实现）。"""

    pool_id: Optional[str]              # 当前所在池；None = 全部为现金
    principal: Decimal                  # 已投入池的本金（计价本位）
    pending_reward: Decimal             # 池内累积未复投奖励
    cash: Decimal                       # 池外现金
    opened_tick: Optional[int]          # 进入当前池的 tick；None=未持仓
    last_compound_tick: Optional[int]   # 最近一次复投的 tick

    def total_value(self) -> Decimal:
        return self.principal + self.pending_reward + self.cash

    @staticmethod
    def empty(initial_cash: Decimal) -> "Position":
        return Position(
            pool_id=None,
            principal=Decimal(0),
            pending_reward=Decimal(0),
            cash=initial_cash,
            opened_tick=None,
            last_compound_tick=None,
        )


@dataclass(frozen=True, slots=True)
class FrictionBreakdown:
    """摩擦成本三分量。total = gas + slippage + lvr。"""

    gas: Decimal
    slippage: Decimal
    lvr: Decimal

    @property
    def total(self) -> Decimal:
        return self.gas + self.slippage + self.lvr


@dataclass(frozen=True, slots=True)
class TradeLog:
    """单次轮动产生的日志条目（最终落 Parquet）。"""

    tick: int
    timestamp: datetime
    operation: OperationType
    from_pool_id: Optional[str]
    to_pool_id: Optional[str]
    amount: Decimal                     # 本次涉及的本金（不含 friction 扣减后的纯进场量）
    gas_cost: Decimal
    slippage_cost: Decimal
    lvr_cost: Decimal
    expected_gain: Decimal
    decision_reason: str                # OK / TAU_FAIL / GATE_FAIL / DATA_ERROR

    @property
    def total_friction(self) -> Decimal:
        return self.gas_cost + self.slippage_cost + self.lvr_cost


@dataclass(frozen=True, slots=True)
class ReinvestLog:
    """单次复投的日志条目。"""

    tick: int
    timestamp: datetime
    pool_id: str
    reward_compounded: Decimal          # 注入本金的浮盈数额
    gas_cost: Decimal
    expected_gain: Decimal              # 复投后下一窗口预期增量收益


# =================================================================
# 轮动决策值对象
# =================================================================

class RotationState(Enum):
    """RotationEngine 内部状态机。"""

    IDLE = "IDLE"
    SCORING = "SCORING"
    RANKED = "RANKED"
    EVALUATING = "EVALUATING"
    COMMITTING = "COMMITTING"
    HOLDING = "HOLDING"
    ERROR = "ERROR"


class DecisionType(Enum):
    HOLD = "HOLD"
    ROTATE = "ROTATE"


class HoldReason(Enum):
    NO_CANDIDATES = "NO_CANDIDATES"
    SAME_POOL = "SAME_POOL"             # top-1 与当前持仓相同
    TAU_FAIL = "TAU_FAIL"               # 偏离度未达 τ
    GATE_FAIL = "GATE_FAIL"             # 净增益未跨过门槛
    DATA_ERROR = "DATA_ERROR"           # 异常 fallback


@dataclass(frozen=True, slots=True)
class RotationDecision:
    """evaluate() 的输出。commit() 仅对 decision_type=ROTATE 生效。"""

    tick: int
    decision_type: DecisionType
    target_pool_id: Optional[str]
    expected_gain: Decimal
    estimated_friction: FrictionBreakdown
    threshold_required: Decimal         # 门槛绝对值（threshold × principal）
    reason: Optional[HoldReason]        # HOLD 时记录原因；ROTATE 时为 None
    notes: str = ""                     # 自由文本，便于排查


@dataclass(frozen=True, slots=True)
class ReinvestDecision:
    """ReinvestEngine.evaluate() 的输出。"""

    tick: int
    do_reinvest: bool
    pending_reward: Decimal
    gas_cost: Decimal
    expected_gain: Decimal
    reason: str                         # OK / NO_REWARDS / NO_POSITION / NEGATIVE_NET


# =================================================================
# 抽象接口
# =================================================================

class IScorer(ABC):
    """评分器抽象接口。实现必须保证纯函数语义。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Scorer 唯一标识，对应 WeightConfig 的 key。"""

    @abstractmethod
    def score(self, snapshot: AssetSnapshot, params: ScoringParams) -> ScoreVector:
        """对 snapshot 中所有池打分，返回 ScoreVector。"""


class IFrictionEstimator(ABC):
    """摩擦成本估算器接口。

    Phase 2 用 stub 实现喂给 RotationEngine；Phase 3 在 backtest/cost_model.py
    提供真实实现（Gas + 滑点 + LVR 三分量）。
    """

    @abstractmethod
    def estimate(
        self,
        op_type: OperationType,
        amount: Decimal,
        pool_id: str,
        snapshot: AssetSnapshot,
    ) -> FrictionBreakdown:
        """对单次操作估算 gas / slippage / lvr 三分量。"""


class IGainEstimator(ABC):
    """期望增量收益估算器接口。"""

    @abstractmethod
    def expected_rotation_gain(
        self,
        position: Position,
        target_pool_id: str,
        snapshot: AssetSnapshot,
        horizon_ticks: int,
    ) -> Decimal:
        """估算「从当前持仓切到 target 后，未来 horizon_ticks 内的增量收益」。"""

    @abstractmethod
    def expected_reinvest_gain(
        self,
        position: Position,
        snapshot: AssetSnapshot,
        horizon_ticks: int,
    ) -> Decimal:
        """估算「将 pending_reward 复投后，未来 horizon_ticks 内的新增收益」。"""


class IRotationPolicy(ABC):
    """轮动决策策略抽象接口（保留为未来扩展点；当前 RotationEngine 直接内联策略）。"""

    @abstractmethod
    def decide(self, position, candidates):
        ...
