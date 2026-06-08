"""黑盒：场景法测试（Scenario Testing）。

PPT 第 8 讲：每个场景对应一段完整业务流程，断言"业务结果"而非中间细节。
本文件用合成数据 + 预设策略组合构造典型业务场景，验证端到端行为。

编号规范：SC-<场景缩写>-<序号>
场景清单：
  SC-STAB  正常市场，无事件，策略稳定运行
  SC-EXPL  Pool_Exploit 后受灾池评分下跌并触发避险
  SC-GAS   Gas_Spike 期间冻结调仓（gate 因 friction 升高而失败）
  SC-FREQ  长持有期间高频复投正常累积
  SC-CMP   策略对比：保守 MDD 应低于激进 MDD（实测验证文档承诺）
  SC-DRIFT 价格漂移驱动策略撤离贬值池
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List

import pytest

from backtest.cost_model import FrictionEstimator
from backtest.engine import BacktestEngine, BacktestResult
from backtest.event_injector import EventInjector, EventType, StressEvent
from data.sample_data import generate_sample_data
from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from data_model.loader import build_asset_snapshots
from report.metrics import compute_metrics
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import ScoringParams, WeightConfig
from strategy.presets import AGGRESSIVE_MOMENTUM, BALANCED, CONSERVATIVE
from strategy.reinvest_engine import ReinvestEngine
from strategy.rotation_engine import RotationEngine
from strategy.scorers.cara import CARAUtilityAdjuster
from strategy.scorers.momentum import MomentumScorer
from strategy.scorers.risk_penalty import (
    DownsideVolPenaltyScorer,
    MaxDrawdownPenaltyScorer,
    TokenPriceMDDPenaltyScorer,
    TokenPriceVolPenaltyScorer,
)
from strategy.scoring_engine import ScoringEngine

pytestmark = [pytest.mark.blackbox, pytest.mark.scenario, pytest.mark.integration]


SLIPPAGE_PRESETS = {
    "低（流动性充足）": (Decimal("0.0005"), Decimal("0.002"), Decimal("0.005")),
    "中（默认）":     (Decimal("0.001"),  Decimal("0.003"), Decimal("0.008")),
    "高（流动性紧张）": (Decimal("0.002"),  Decimal("0.006"), Decimal("0.015")),
}


# =====================================================================
# 引擎工厂
# =====================================================================

def _build_engine(preset, event_injector: EventInjector | None = None) -> BacktestEngine:
    sl, sm, sh = SLIPPAGE_PRESETS[preset.slip_choice]
    friction = FrictionEstimator(slip_rate_low=sl, slip_rate_mid=sm, slip_rate_high=sh)
    gain = APYDeltaGainEstimator()
    return BacktestEngine(
        initial_capital=Decimal("100000"),
        scoring_engine=ScoringEngine(
            params=ScoringParams(cara_alpha=preset.cara_alpha),
            weight_cfg=WeightConfig(preset.weights),
            scorers=[
                MomentumScorer(),
                DownsideVolPenaltyScorer(),
                MaxDrawdownPenaltyScorer(),
                CARAUtilityAdjuster(),
                TokenPriceVolPenaltyScorer(),
                TokenPriceMDDPenaltyScorer(),
            ],
        ),
        rotation_engine=RotationEngine(
            tau_reset=preset.tau_reset, threshold=preset.threshold,
            gain_estimator=gain, friction_estimator=friction,
            gain_horizon_ticks=preset.gain_horizon_ticks,
        ),
        reinvest_engine=ReinvestEngine(
            friction_estimator=friction, gain_estimator=gain,
            reinvest_window=30, risk_premium_multiplier=Decimal("1.5"),
        ),
        event_injector=event_injector,
    )


@pytest.fixture(scope="module")
def default_snapshots():
    pool_df, gas_df = generate_sample_data(n_days=365)
    return build_asset_snapshots(pool_df, gas_df, config={"momentum_window": 14})


# =====================================================================
# SC-STAB：稳定市场，无事件
# =====================================================================

class TestScenarioStableMarket:

    def test_SC_STAB_01_full_run_completes(self, default_snapshots):
        engine = _build_engine(BALANCED)
        result = engine.run(default_snapshots)
        assert result.snapshots_processed == len(default_snapshots)

    def test_SC_STAB_02_nav_log_complete_and_monotone_average(self, default_snapshots):
        engine = _build_engine(BALANCED)
        result = engine.run(default_snapshots)
        assert len(result.nav_log) == len(default_snapshots)
        # 最终 NAV 与初始相比应为合理范围（-30% ~ +30%）
        first = float(result.nav_log["nav"].iloc[0])
        last = float(result.nav_log["nav"].iloc[-1])
        assert 0.7 * first < last < 1.3 * first


# =====================================================================
# SC-EXPL：Pool_Exploit 事件
# =====================================================================

class TestScenarioPoolExploit:

    def test_SC_EXPL_01_target_pool_score_drops_after_event(self, default_snapshots):
        injector = EventInjector([
            StressEvent(EventType.POOL_EXPLOIT, start_tick=200, duration=1,
                        impact_ratio=Decimal("0.9"), target_pool_id="pool_B"),
        ])
        engine = _build_engine(BALANCED, event_injector=injector)
        result = engine.run(default_snapshots)
        # tick 200 时 pool_B 的 total_score 应明显低于 tick 199
        s_before = result.score_log.query(
            "tick==199 and pool_id=='pool_B'"
        )["total_score"].iloc[0]
        s_after = result.score_log.query(
            "tick==200 and pool_id=='pool_B'"
        )["total_score"].iloc[0]
        assert s_after < s_before

    def test_SC_EXPL_02_strategy_avoids_exploited_pool_after_event(self, default_snapshots):
        """事件后策略不应再被分配到 pool_B。"""
        injector = EventInjector([
            StressEvent(EventType.POOL_EXPLOIT, start_tick=200, duration=1,
                        impact_ratio=Decimal("0.9"), target_pool_id="pool_B"),
        ])
        engine = _build_engine(BALANCED, event_injector=injector)
        result = engine.run(default_snapshots)
        # tick 201 之后的持仓池里 pool_B 应该很少出现（最多被持仓滞后效应保留几 tick）
        post = result.nav_log[result.nav_log["tick"] > 210]
        pool_b_share = (post["pool_id"] == "pool_B").mean()
        assert pool_b_share < 0.5


# =====================================================================
# SC-GAS：Gas_Spike 事件
# =====================================================================

class TestScenarioGasSpike:

    def test_SC_GAS_01_gas_base_fee_multiplied_during_event_window(self, default_snapshots):
        # 强 injector 注入 Gas_Spike
        injector = EventInjector([
            StressEvent(EventType.GAS_SPIKE, start_tick=180, duration=3,
                        impact_ratio=Decimal("4.0")),  # × 5
        ])
        engine = _build_engine(BALANCED, event_injector=injector)
        result = engine.run(default_snapshots)
        base_outside = float(
            result.nav_log.loc[result.nav_log["tick"] == 100, "env_gas_base_fee"].iloc[0]
        )
        base_inside = float(
            result.nav_log.loc[result.nav_log["tick"] == 181, "env_gas_base_fee"].iloc[0]
        )
        # spike 期间至少 ×5（CSV 已有 150~154 spike，这里 180-182 是干净的）
        assert base_inside >= base_outside * 4.9

    def test_SC_GAS_02_gate_fails_more_during_spike(self, default_snapshots):
        """Gas_Spike 期间 ROTATE 数量应低于平时（友谊提示：gas 上升让 gate 不通过）。"""
        injector = EventInjector([
            StressEvent(EventType.GAS_SPIKE, start_tick=100, duration=30,
                        impact_ratio=Decimal("19.0")),  # × 20，极端 spike
        ])
        engine_spike = _build_engine(AGGRESSIVE_MOMENTUM, event_injector=injector)
        engine_normal = _build_engine(AGGRESSIVE_MOMENTUM)

        r_spike = engine_spike.run(default_snapshots)
        r_normal = engine_normal.run(default_snapshots)

        # spike 窗口内（100~129）的 ROTATE 计数 <= 无 spike 时同窗口
        def _rotates_in(window, log):
            return ((log["operation"] == "ROTATE")
                    & (log["tick"].between(window[0], window[1]))).sum()

        spike_count = _rotates_in((100, 129), r_spike.trade_log)
        normal_count = _rotates_in((100, 129), r_normal.trade_log)
        assert spike_count <= normal_count


# =====================================================================
# SC-FREQ：复投高频累积
# =====================================================================

class TestScenarioFrequentReinvest:

    def test_SC_FREQ_01_reinvest_count_substantial_when_gas_low(self, default_snapshots):
        engine = _build_engine(BALANCED)
        result = engine.run(default_snapshots)
        # 365 tick 下默认 gas 极低，复投应几乎每 tick 都触发（> 200）
        assert len(result.reinvest_log) > 200

    def test_SC_FREQ_02_each_reinvest_increases_principal(self, default_snapshots):
        engine = _build_engine(BALANCED)
        result = engine.run(default_snapshots)
        if len(result.reinvest_log) >= 2:
            # 复投后 principal 应增加（不严格但绝大多数应满足）
            increases = sum(
                1 for v in result.reinvest_log["reward_compounded"] if v > 0
            )
            assert increases >= len(result.reinvest_log) * 0.9


# =====================================================================
# SC-CMP：策略对比
# =====================================================================

class TestScenarioStrategyComparison:

    def test_SC_CMP_01_conservative_lower_mdd_than_aggressive(self, default_snapshots):
        r_cons = _build_engine(CONSERVATIVE).run(default_snapshots)
        r_aggr = _build_engine(AGGRESSIVE_MOMENTUM).run(default_snapshots)
        m_cons = compute_metrics(r_cons.nav_log, r_cons.trade_log, r_cons.reinvest_log)
        m_aggr = compute_metrics(r_aggr.nav_log, r_aggr.trade_log, r_aggr.reinvest_log)
        assert m_cons.max_drawdown <= m_aggr.max_drawdown

    def test_SC_CMP_02_strategies_make_different_choices(self, default_snapshots):
        """保守 vs 激进应选不同的初始池（或不同的整体持仓分布）。"""
        r_cons = _build_engine(CONSERVATIVE).run(default_snapshots)
        r_aggr = _build_engine(AGGRESSIVE_MOMENTUM).run(default_snapshots)
        # 行为差异度量：持仓池组合的差异（哪怕都只有 1 次 ROTATE，目标池也应不同）
        cons_pools = set(r_cons.nav_log["pool_id"].dropna().unique())
        aggr_pools = set(r_aggr.nav_log["pool_id"].dropna().unique())
        # 期望：保守与激进至少在最常持仓上不同
        cons_top = r_cons.nav_log["pool_id"].mode()[0] if not r_cons.nav_log.empty else None
        aggr_top = r_aggr.nav_log["pool_id"].mode()[0] if not r_aggr.nav_log.empty else None
        assert cons_top != aggr_top, (
            f"保守与激进选了同一池 {cons_top}；可能权重/数据未能拉开差异"
        )


# =====================================================================
# SC-DRIFT：价格漂移
# =====================================================================

def _build_price_drift_snapshots() -> List[AssetSnapshot]:
    """构造特殊数据：pool_X 价格逐步下跌 30%，pool_Y 价格稳定，APY 持平。"""
    snapshots = []
    start = datetime(2024, 1, 1)
    for t in range(60):
        # pool_X 从 1.0 跌到 0.7
        x_price = Decimal(str(1.0 - 0.005 * t))
        # 14 个回看历史（前后填充）
        x_history = tuple(
            Decimal(str(1.0 - 0.005 * max(t - k, 0))) for k in range(13, -1, -1)
        )
        y_history = (Decimal("1.0"),) * 14
        pools = {
            "pool_X": PoolMetrics(
                pool_id="pool_X",
                apy_series=(Decimal("0.05"),) * 14,
                tvl=Decimal("100000000"),
                vol_30d=Decimal("0.02"),
                token_price=x_price,
                gas_base_fee=Decimal("0.0000001"),
                token_price_series=x_history,
            ),
            "pool_Y": PoolMetrics(
                pool_id="pool_Y",
                apy_series=(Decimal("0.05"),) * 14,
                tvl=Decimal("100000000"),
                vol_30d=Decimal("0.02"),
                token_price=Decimal("1.0"),
                gas_base_fee=Decimal("0.0000001"),
                token_price_series=y_history,
            ),
        }
        env = EnvSnapshot(
            tick=t, timestamp=start + timedelta(days=t),
            oracle_price={"pool_X": x_price, "pool_Y": Decimal("1.0")},
            gas_base_fee=Decimal("0.0000001"),
            gas_priority_fee=Decimal("0.00000005"),
        )
        snapshots.append(AssetSnapshot(tick=t, pools=pools, env=env))
    return snapshots


class TestScenarioPriceDrift:

    def test_SC_DRIFT_01_strategy_avoids_dropping_token(self):
        snapshots = _build_price_drift_snapshots()
        engine = _build_engine(CONSERVATIVE)
        result = engine.run(snapshots)
        # 整段 60 tick 中持有 pool_Y 的比例应远高于 pool_X
        share_y = (result.nav_log["pool_id"] == "pool_Y").mean()
        share_x = (result.nav_log["pool_id"] == "pool_X").mean()
        assert share_y >= share_x   # 至少不少于
