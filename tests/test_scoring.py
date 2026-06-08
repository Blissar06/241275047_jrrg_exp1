"""Phase 2 第一段：评分层单元测试。

覆盖：
  - 各 Scorer 输出确定性（NFR-02）
  - 重复注册抛 DuplicateScorerError（E-RT-003）
  - vol=0 / 单池等退化场景的鲁棒性
  - WeightConfig 自动归一化
  - 同分排序的稳定性
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict, Tuple

import pytest

from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from strategy.interfaces import (
    DuplicateScorerError,
    PoolScore,
    ScoreVector,
    ScoringContext,
    ScoringParams,
    WeightConfig,
)
from strategy.scorers._common import zscore_dict
from strategy.scorers.cara import CARAUtilityAdjuster, cara_utility
from strategy.scorers.momentum import MomentumScorer, ewma
from strategy.scorers.risk_penalty import (
    DownsideVolPenaltyScorer,
    MaxDrawdownPenaltyScorer,
    downside_volatility,
    max_drawdown,
)
from strategy.scoring_engine import ScoringEngine

import numpy as np


# ============================================================
# Helpers
# ============================================================

def _make_pool(pool_id: str, apy_series: Tuple[Decimal, ...], tvl: str = "1000000") -> PoolMetrics:
    return PoolMetrics(
        pool_id=pool_id,
        apy_series=apy_series,
        tvl=Decimal(tvl),
        vol_30d=Decimal("0.02"),
        token_price=Decimal("1.0"),
        gas_base_fee=Decimal("20"),
    )


def _make_snapshot(pools: Dict[str, PoolMetrics], tick: int = 0) -> AssetSnapshot:
    env = EnvSnapshot(
        tick=tick,
        timestamp=datetime(2024, 1, 1),
        oracle_price={pid: Decimal("1.0") for pid in pools},
        gas_base_fee=Decimal("20"),
        gas_priority_fee=Decimal("1.5"),
    )
    return AssetSnapshot(tick=tick, pools=pools, env=env)


def _ramp(start: float, step: float, n: int) -> Tuple[Decimal, ...]:
    return tuple(Decimal(str(start + step * i)) for i in range(n))


# ============================================================
# Helper functions (low-level)
# ============================================================

def test_ewma_recovers_constant_series():
    s = (Decimal("0.05"),) * 10
    assert ewma(s, Decimal("0.85")) == Decimal("0.05")


def test_ewma_lambda_out_of_range_raises():
    with pytest.raises(ValueError):
        ewma((Decimal("0.05"),), Decimal(1))


def test_downside_vol_zero_when_monotone_increase():
    arr = np.array([0.05, 0.06, 0.07, 0.08])
    assert downside_volatility(arr) == 0.0


def test_max_drawdown_recovers_simple_case():
    # 0.10 → 0.05 → 0.08，峰值 0.10，最低回撤 = 1 - 0.05/0.10 = 0.5
    arr = np.array([0.10, 0.05, 0.08])
    assert max_drawdown(arr) == pytest.approx(0.5)


def test_zscore_dict_all_equal_returns_zero():
    out = zscore_dict({"a": Decimal("0.05"), "b": Decimal("0.05")})
    assert out["a"] == Decimal(0)
    assert out["b"] == Decimal(0)


def test_cara_utility_monotone_increasing_in_r():
    u_low = cara_utility(0.01, alpha=2.0)
    u_high = cara_utility(0.10, alpha=2.0)
    assert u_high > u_low


# ============================================================
# Scorer determinism (NFR-02)
# ============================================================

def test_momentum_scorer_deterministic():
    pools = {
        "a": _make_pool("a", _ramp(0.05, 0.001, 14)),
        "b": _make_pool("b", _ramp(0.06, 0.001, 14)),
        "c": _make_pool("c", _ramp(0.04, 0.001, 14)),
    }
    snap = _make_snapshot(pools)
    params = ScoringParams()
    s1 = MomentumScorer().score(snap, params)
    s2 = MomentumScorer().score(snap, params)
    assert s1.scores == s2.scores
    # b 应该 > a > c（apy 水平排序）
    assert s1.scores["b"] > s1.scores["a"] > s1.scores["c"]


def test_vol_penalty_higher_volatility_lower_score():
    stable = (Decimal("0.05"),) * 30
    volatile = tuple(Decimal(str(0.05 + ((-1) ** i) * 0.02)) for i in range(30))
    pools = {
        "stable": _make_pool("stable", stable),
        "volatile": _make_pool("volatile", volatile),
    }
    snap = _make_snapshot(pools)
    sv = DownsideVolPenaltyScorer().score(snap, ScoringParams())
    assert sv.scores["stable"] > sv.scores["volatile"]


def test_mdd_scorer_high_drawdown_lower_score():
    flat = (Decimal("0.05"),) * 30
    crash_series = tuple(Decimal("0.10") for _ in range(15)) + tuple(Decimal("0.02") for _ in range(15))
    pools = {
        "flat": _make_pool("flat", flat),
        "crash": _make_pool("crash", crash_series),
    }
    snap = _make_snapshot(pools)
    sv = MaxDrawdownPenaltyScorer().score(snap, ScoringParams())
    assert sv.scores["flat"] > sv.scores["crash"]


def test_cara_scorer_high_apy_higher_score():
    pools = {
        "low": _make_pool("low", (Decimal("0.03"),) * 5),
        "high": _make_pool("high", (Decimal("0.12"),) * 5),
    }
    snap = _make_snapshot(pools)
    sv = CARAUtilityAdjuster().score(snap, ScoringParams(cara_alpha=Decimal("2.0")))
    assert sv.scores["high"] > sv.scores["low"]


def test_all_scorers_handle_single_pool_without_error():
    """单池场景 std=0，z-score 应返回 0 而非崩溃（E-RT-002 边界）。"""
    pools = {"only": _make_pool("only", _ramp(0.05, 0.001, 14))}
    snap = _make_snapshot(pools)
    params = ScoringParams()
    for scorer in [
        MomentumScorer(),
        DownsideVolPenaltyScorer(),
        MaxDrawdownPenaltyScorer(),
        CARAUtilityAdjuster(),
    ]:
        sv = scorer.score(snap, params)
        assert sv.scores["only"] == Decimal(0)


# ============================================================
# ScoringEngine
# ============================================================

def test_engine_register_duplicate_raises():
    engine = ScoringEngine(
        params=ScoringParams(),
        weight_cfg=WeightConfig({"momentum": Decimal("1.0")}),
    )
    engine.register(MomentumScorer())
    with pytest.raises(DuplicateScorerError):
        engine.register(MomentumScorer())


def test_engine_run_without_scorers_raises():
    engine = ScoringEngine(
        params=ScoringParams(),
        weight_cfg=WeightConfig({"momentum": Decimal("1.0")}),
    )
    pools = {"a": _make_pool("a", _ramp(0.05, 0.001, 5))}
    with pytest.raises(RuntimeError):
        engine.run(_make_snapshot(pools))


def test_engine_weights_normalized():
    cfg = WeightConfig({
        "momentum": Decimal("2.0"),
        "vol_penalty": Decimal("1.0"),
        "mdd_penalty": Decimal("1.0"),
    })
    engine = ScoringEngine(params=ScoringParams(), weight_cfg=cfg)
    total = sum(engine.weight_cfg.weights.values())
    assert total == Decimal(1)


def test_weight_cfg_zero_total_raises():
    cfg = WeightConfig({"x": Decimal(0), "y": Decimal(0)})
    with pytest.raises(ValueError):
        cfg.normalized()


def test_engine_full_pipeline_rankings_make_sense():
    pools = {
        "best": _make_pool("best", _ramp(0.08, 0.002, 30)),     # 高 APY、上升
        "mid":  _make_pool("mid", (Decimal("0.05"),) * 30),     # 平稳
        "worst": _make_pool("worst", tuple(Decimal("0.10") if i < 15 else Decimal("0.02") for i in range(30))),  # 高回撤
    }
    snap = _make_snapshot(pools)

    engine = ScoringEngine(
        params=ScoringParams(),
        weight_cfg=WeightConfig({
            "momentum": Decimal("0.40"),
            "vol_penalty": Decimal("0.25"),
            "mdd_penalty": Decimal("0.20"),
            "cara": Decimal("0.15"),
        }),
        scorers=[
            MomentumScorer(),
            DownsideVolPenaltyScorer(),
            MaxDrawdownPenaltyScorer(),
            CARAUtilityAdjuster(),
        ],
    )

    table = engine.run(snap)
    assert len(table.rankings) == 3
    # best 应排在第一
    assert table.rankings[0].pool_id == "best"
    # 每个条目带 4 个分量
    assert set(table.rankings[0].components.keys()) == {
        "momentum", "vol_penalty", "mdd_penalty", "cara",
    }


def test_engine_excluded_pools_filtered():
    pools = {
        "a": _make_pool("a", _ramp(0.05, 0.001, 5)),
        "b": _make_pool("b", _ramp(0.06, 0.001, 5)),
    }
    snap = _make_snapshot(pools)
    engine = ScoringEngine(
        params=ScoringParams(),
        weight_cfg=WeightConfig({"momentum": Decimal("1.0")}),
        scorers=[MomentumScorer()],
    )
    table = engine.run(snap, ScoringContext(excluded_pools=("b",)))
    assert {ps.pool_id for ps in table.rankings} == {"a"}


def test_engine_stable_sort_on_ties():
    """同分时按 pool_id 升序，结果跨次稳定（NFR-02）。"""
    pools = {
        "ZZZ": _make_pool("ZZZ", (Decimal("0.05"),) * 5),
        "AAA": _make_pool("AAA", (Decimal("0.05"),) * 5),
        "MMM": _make_pool("MMM", (Decimal("0.05"),) * 5),
    }
    snap = _make_snapshot(pools)
    engine = ScoringEngine(
        params=ScoringParams(),
        weight_cfg=WeightConfig({"momentum": Decimal("1.0")}),
        scorers=[MomentumScorer()],
    )
    table1 = engine.run(snap)
    table2 = engine.run(snap)
    assert [ps.pool_id for ps in table1.rankings] == ["AAA", "MMM", "ZZZ"]
    assert table1.rankings == table2.rankings


def test_engine_top_n_returns_correct_count():
    pools = {f"p{i}": _make_pool(f"p{i}", _ramp(0.04 + i * 0.01, 0.001, 10)) for i in range(5)}
    snap = _make_snapshot(pools)
    engine = ScoringEngine(
        params=ScoringParams(),
        weight_cfg=WeightConfig({"momentum": Decimal("1.0")}),
        scorers=[MomentumScorer()],
    )
    top = engine.top_n(snap, n=3)
    assert len(top) == 3
    # apy 越高分越高，所以应是 p4, p3, p2
    assert [ps.pool_id for ps in top] == ["p4", "p3", "p2"]


def test_engine_run_deterministic():
    """相同输入两次 run() 结果完全一致。"""
    pools = {f"p{i}": _make_pool(f"p{i}", _ramp(0.04 + i * 0.01, 0.001, 30)) for i in range(4)}
    snap = _make_snapshot(pools)
    engine = ScoringEngine(
        params=ScoringParams(),
        weight_cfg=WeightConfig({
            "momentum": Decimal("0.5"),
            "vol_penalty": Decimal("0.3"),
            "mdd_penalty": Decimal("0.2"),
        }),
        scorers=[
            MomentumScorer(),
            DownsideVolPenaltyScorer(),
            MaxDrawdownPenaltyScorer(),
        ],
    )
    t1 = engine.run(snap)
    t2 = engine.run(snap)
    assert t1.rankings == t2.rankings
