"""性能基准测试（pytest-benchmark）。

PPT 第 8 讲提到性能测试 / 压力测试 / 负载测试。本文件按基准（baseline）+ 回归
（regression）思路设计：
  - 每个 benchmark 自动统计 mean / median / stddev / rounds
  - pytest-benchmark 可保存历史 JSON，下次跑会对比；阈值由 --benchmark-compare-fail 设定
  - 跑法：
      pytest tests/perf -m perf
      pytest tests/perf --benchmark-save=baseline
      pytest tests/perf --benchmark-compare=baseline --benchmark-compare-fail=mean:20%

编号规范：PERF-<对象缩写>-<NN>
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from backtest.cost_model import FrictionEstimator
from backtest.engine import BacktestEngine
from data_model.loader import build_asset_snapshots
from data.sample_data import generate_sample_data
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import (
    OperationType,
    Position,
    ScoringContext,
    ScoringParams,
    WeightConfig,
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

pytestmark = [pytest.mark.perf, pytest.mark.slow]


# =====================================================================
# 引擎工厂（与生产配置一致）
# =====================================================================

def _build_engine() -> BacktestEngine:
    friction = FrictionEstimator()
    gain = APYDeltaGainEstimator()
    return BacktestEngine(
        initial_capital=Decimal("100000"),
        scoring_engine=ScoringEngine(
            params=ScoringParams(),
            weight_cfg=WeightConfig({
                "momentum": Decimal("0.30"),
                "vol_penalty": Decimal("0.15"),
                "mdd_penalty": Decimal("0.15"),
                "cara": Decimal("0.10"),
                "price_vol_penalty": Decimal("0.15"),
                "price_mdd_penalty": Decimal("0.15"),
            }),
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
            tau_reset=Decimal("0.05"), threshold=Decimal("0.001"),
            gain_estimator=gain, friction_estimator=friction,
        ),
        reinvest_engine=ReinvestEngine(
            friction_estimator=friction, gain_estimator=gain,
            reinvest_window=30, risk_premium_multiplier=Decimal("1.5"),
        ),
    )


@pytest.fixture(scope="module")
def snaps_365_3():
    pool_df, gas_df = generate_sample_data(n_days=365)
    return build_asset_snapshots(pool_df, gas_df, config={"momentum_window": 14})


@pytest.fixture(scope="module")
def snaps_1000_5():
    pool_df, gas_df = generate_sample_data(
        n_days=1000,
        pool_ids=[f"p{i}" for i in range(5)],
        crash_pool_index=1,
    )
    return build_asset_snapshots(pool_df, gas_df, config={"momentum_window": 14})


# =====================================================================
# 端到端基准
# =====================================================================

@pytest.mark.benchmark(group="engine")
def test_PERF_RUN_01_run_365t_3pools(benchmark, snaps_365_3):
    """365 tick × 3 池：典型回测，目标 < 1s。"""
    def _run():
        return _build_engine().run(snaps_365_3)

    result = benchmark(_run)
    assert result.snapshots_processed == 365


@pytest.mark.benchmark(group="engine")
def test_PERF_RUN_02_run_1000t_5pools(benchmark, snaps_1000_5):
    """1000 tick × 5 池：中等规模，目标 < 5s。"""
    def _run():
        return _build_engine().run(snaps_1000_5)

    result = benchmark.pedantic(_run, rounds=3, iterations=1)
    assert result.snapshots_processed == 1000


# =====================================================================
# 单组件基准
# =====================================================================

@pytest.fixture
def scoring_engine():
    engine = _build_engine()
    return engine.scoring


@pytest.mark.benchmark(group="scoring")
def test_PERF_SC_01_scoring_engine_run(benchmark, snaps_365_3, scoring_engine):
    """ScoringEngine.run 单次：单 tick 全部 scorer 跑完一遍。"""
    snap = snaps_365_3[len(snaps_365_3) // 2]
    benchmark(scoring_engine.run, snap, ScoringContext())


@pytest.mark.benchmark(group="cost")
def test_PERF_FR_01_friction_estimate_rotate(benchmark, snaps_365_3):
    """FrictionEstimator.estimate ROTATE 单次。"""
    fe = FrictionEstimator()
    snap = snaps_365_3[100]
    pool_id = next(iter(snap.pools.keys()))
    benchmark(fe.estimate, OperationType.ROTATE, Decimal("50000"), pool_id, snap)


# =====================================================================
# Scorer 单体基准（找性能瓶颈）
# =====================================================================

@pytest.mark.benchmark(group="scorers")
@pytest.mark.parametrize("scorer_cls", [
    MomentumScorer,
    DownsideVolPenaltyScorer,
    MaxDrawdownPenaltyScorer,
    CARAUtilityAdjuster,
    TokenPriceVolPenaltyScorer,
    TokenPriceMDDPenaltyScorer,
])
def test_PERF_SCORER_01_individual_scorer(benchmark, snaps_365_3, scorer_cls):
    scorer = scorer_cls()
    snap = snaps_365_3[100]
    params = ScoringParams()
    benchmark(scorer.score, snap, params)
