"""Phase 8 回归测试：MTM + 价格风险评分 + presets。

核心断言：
  - MTM 正确按 token_price 比例重估持仓
  - TokenPriceVolPenaltyScorer / TokenPriceMDDPenaltyScorer 按预期方向打分
  - APYDeltaGainEstimator 在 drift 模式下把价格漂移并入 effective APY
  - 端到端：保守策略在合成数据上的 MDD < 激进策略
  - 全部 5 个预设跑通且 metrics 字段合理
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List

import pytest

from backtest.cost_model import FrictionEstimator
from backtest.engine import BacktestEngine
from backtest.event_injector import EventInjector
from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from data_model.loader import build_asset_snapshots
from data.sample_data import generate_sample_data
from report.attribution import compute_attribution
from report.metrics import compute_metrics
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import Position, ScoringParams, WeightConfig
from strategy.presets import (
    AGGRESSIVE_MOMENTUM,
    ALL_PRESETS,
    BALANCED,
    CONSERVATIVE,
    StrategyPreset,
    list_preset_names,
)
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


SLIPPAGE_PRESETS = {
    "低（流动性充足）": (Decimal("0.0005"), Decimal("0.002"), Decimal("0.005")),
    "中（默认）":     (Decimal("0.001"),  Decimal("0.003"), Decimal("0.008")),
    "高（流动性紧张）": (Decimal("0.002"),  Decimal("0.006"), Decimal("0.015")),
}


def _make_engine(preset: StrategyPreset) -> BacktestEngine:
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
        event_injector=None,
    )


# =================================================================
# Mark-to-market
# =================================================================

def _pool_with_price(pid: str, apy: str, token_price: str,
                    token_history: tuple = ()) -> PoolMetrics:
    return PoolMetrics(
        pool_id=pid,
        apy_series=(Decimal(apy),) * 14,
        tvl=Decimal("100000000"),
        vol_30d=Decimal("0.02"),
        token_price=Decimal(token_price),
        gas_base_fee=Decimal("0.0000001"),
        token_price_series=token_history,
    )


def _snap(pools, tick=0, base_fee="0.0000001"):
    env = EnvSnapshot(
        tick=tick, timestamp=datetime(2024, 1, 1) + timedelta(days=tick),
        oracle_price={pid: pools[pid].token_price for pid in pools},
        gas_base_fee=Decimal(base_fee),
        gas_priority_fee=Decimal("0.00000005"),
    )
    return AssetSnapshot(tick=tick, pools=pools, env=env)


def test_mark_to_market_scales_principal_by_price_ratio():
    """显式构造 prev/curr snap，验证 MTM 把 principal * 0.9 当 token 跌 10%。"""
    pools_t0 = {"x": _pool_with_price("x", "0.05", "1.00")}
    pools_t1 = {"x": _pool_with_price("x", "0.05", "0.90")}    # -10%
    snap0 = _snap(pools_t0, tick=0)
    snap1 = _snap(pools_t1, tick=1)

    engine = _make_engine(BALANCED)
    pos = Position(
        pool_id="x", principal=Decimal("100000"),
        pending_reward=Decimal("500"), cash=Decimal("0"),
        opened_tick=0, last_compound_tick=0,
    )
    new_pos = engine._mark_to_market(pos, snap0, snap1)
    assert new_pos.principal == Decimal("90000")
    assert new_pos.pending_reward == Decimal("450")
    # cash 不受 MTM 影响
    assert new_pos.cash == pos.cash


def test_mark_to_market_no_op_when_no_position():
    pools_t0 = {"x": _pool_with_price("x", "0.05", "1.00")}
    pools_t1 = {"x": _pool_with_price("x", "0.05", "0.90")}
    engine = _make_engine(BALANCED)
    cash_pos = Position.empty(initial_cash=Decimal("100000"))
    out = engine._mark_to_market(cash_pos, _snap(pools_t0), _snap(pools_t1, tick=1))
    assert out == cash_pos


def test_mark_to_market_no_op_when_ratio_one():
    pools = {"x": _pool_with_price("x", "0.05", "1.00")}
    engine = _make_engine(BALANCED)
    pos = Position(
        pool_id="x", principal=Decimal("100000"),
        pending_reward=Decimal(0), cash=Decimal(0),
        opened_tick=0, last_compound_tick=0,
    )
    out = engine._mark_to_market(pos, _snap(pools), _snap(pools, tick=1))
    assert out is pos


# =================================================================
# 价格风险评分器
# =================================================================

def test_price_vol_penalty_lower_for_stable_pool():
    """稳定 token (常数 price) 应获得高分；波动 token 应获得低分。"""
    stable_series = (Decimal("1.0"),) * 30
    volatile_series = tuple(
        Decimal(str(1.0 + ((-1) ** i) * 0.05)) for i in range(30)
    )
    pools = {
        "stable":   _pool_with_price("stable", "0.05", "1.00", stable_series),
        "volatile": _pool_with_price("volatile", "0.05", "1.00", volatile_series),
    }
    snap = _snap(pools)
    sv = TokenPriceVolPenaltyScorer().score(snap, ScoringParams())
    assert sv.scores["stable"] > sv.scores["volatile"]


def test_price_mdd_penalty_lower_for_drawdown_pool():
    """有过价格大跌的 token 应被打低分。"""
    flat_series = (Decimal("1.0"),) * 30
    crash_series = (Decimal("1.0"),) * 15 + (Decimal("0.5"),) * 15
    pools = {
        "flat":  _pool_with_price("flat", "0.05", "1.00", flat_series),
        "crash": _pool_with_price("crash", "0.05", "0.50", crash_series),
    }
    snap = _snap(pools)
    sv = TokenPriceMDDPenaltyScorer().score(snap, ScoringParams())
    assert sv.scores["flat"] > sv.scores["crash"]


def test_price_scorers_handle_empty_series():
    """token_price_series 为空 → raw=0 → z-score=0，不抛错。"""
    pools = {"a": _pool_with_price("a", "0.05", "1.00")}  # 无 history
    snap = _snap(pools)
    for scorer in (TokenPriceVolPenaltyScorer(), TokenPriceMDDPenaltyScorer()):
        sv = scorer.score(snap, ScoringParams())
        assert sv.scores["a"] == Decimal(0)


# =================================================================
# GainEstimator 漂移并入
# =================================================================

def test_gain_estimator_incorporates_negative_price_drift():
    """target 的近期价格在跌 → effective APY < raw APY → expected_gain 减少。"""
    # 14 天连续下跌 1%/天 → ~年化 -365%
    decline = tuple(Decimal(str(1.0 - 0.01 * i)) for i in range(14))
    flat = (Decimal("1.0"),) * 14
    pools = {
        "good":  _pool_with_price("good", "0.05", "0.87", decline),
        "stable": _pool_with_price("stable", "0.05", "1.00", flat),
    }
    snap = _snap(pools, tick=14)
    pos = Position(
        pool_id="stable", principal=Decimal("100000"),
        pending_reward=Decimal(0), cash=Decimal(0),
        opened_tick=0, last_compound_tick=0,
    )
    drift_aware = APYDeltaGainEstimator(use_price_drift=True)
    pure_apy = APYDeltaGainEstimator(use_price_drift=False)

    g_drift = drift_aware.expected_rotation_gain(pos, "good", snap, 30)
    g_pure = pure_apy.expected_rotation_gain(pos, "good", snap, 30)
    # 同 APY 时 pure 返回 0；drift 看到 good 在跌 → 返回负值
    assert g_drift < g_pure
    assert g_drift < 0


# =================================================================
# 端到端：preset 在合成数据上的差异化表现
# =================================================================

@pytest.fixture(scope="module")
def synthetic_snapshots():
    pool_df, gas_df = generate_sample_data()
    return build_asset_snapshots(pool_df, gas_df, config={"momentum_window": 14})


def test_conservative_has_lower_mdd_than_aggressive(synthetic_snapshots):
    """价格风险敏感的策略应在合成数据上跑出更低的 MDD。"""
    r_conserv = _make_engine(CONSERVATIVE).run(synthetic_snapshots)
    r_aggr = _make_engine(AGGRESSIVE_MOMENTUM).run(synthetic_snapshots)
    m_cons = compute_metrics(r_conserv.nav_log, r_conserv.trade_log, r_conserv.reinvest_log)
    m_aggr = compute_metrics(r_aggr.nav_log, r_aggr.trade_log, r_aggr.reinvest_log)
    assert m_cons.max_drawdown < m_aggr.max_drawdown


def test_all_presets_produce_nonzero_drawdown_or_meaningful_run(synthetic_snapshots):
    """合成数据已含价格波动，至少有一个 preset 应跑出 MDD > 0；
    所有 preset 都至少完成全部 tick。"""
    nonzero_mdds = 0
    for preset in ALL_PRESETS:
        result = _make_engine(preset).run(synthetic_snapshots)
        assert result.snapshots_processed == len(synthetic_snapshots)
        m = compute_metrics(result.nav_log, result.trade_log, result.reinvest_log)
        if m.max_drawdown > Decimal(0):
            nonzero_mdds += 1
    assert nonzero_mdds >= 3, (
        f"只有 {nonzero_mdds} 个 preset 出现非零 MDD；预期价格波动应让大多数 preset 经历回撤"
    )


def test_sharpe_within_reasonable_range(synthetic_snapshots):
    """Sharpe 应该在 [-3, 10] 内（不是之前那种 1198 的伪值）。"""
    for preset in ALL_PRESETS:
        result = _make_engine(preset).run(synthetic_snapshots)
        m = compute_metrics(result.nav_log, result.trade_log, result.reinvest_log)
        sharpe = float(m.sharpe_ratio)
        assert -3 < sharpe < 10, f"{preset.name} Sharpe={sharpe} 超出合理区间"


# =================================================================
# Presets 注册表
# =================================================================

def test_all_5_preset_names_registered():
    names = list_preset_names()
    assert len(names) == 5
    assert set(names) == {
        "均衡（默认）", "保守稳健", "激进动量", "低频价值", "极端风险厌恶",
    }


def test_preset_weights_sum_positive():
    """所有 preset 的权重和必须 > 0 才能归一化。"""
    for p in ALL_PRESETS:
        total = sum(p.weights.values())
        assert total > Decimal(0), f"{p.name} 权重和={total}"
