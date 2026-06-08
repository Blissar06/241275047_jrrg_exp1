"""Phase 4 命令 4-2：attribution 单元测试。"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from report.attribution import (
    AttributionReport,
    compute_attribution,
    theoretical_nav_path,
)


# ========== theoretical_nav_path ==========

def test_theoretical_path_compounds_max_apy_each_tick():
    score_log = pd.DataFrame([
        {"tick": 0, "pool_id": "a", "apy": 0.10},
        {"tick": 0, "pool_id": "b", "apy": 0.05},
        {"tick": 1, "pool_id": "a", "apy": 0.10},
        {"tick": 1, "pool_id": "b", "apy": 0.20},
    ])
    path = theoretical_nav_path(score_log, Decimal("100"), periods_per_year=365)
    # tick 0 用 max=0.10：100 × (1 + 0.10/365)
    expected_t0 = 100 * (1 + 0.10 / 365)
    expected_t1 = expected_t0 * (1 + 0.20 / 365)
    assert float(path.iloc[0]) == pytest.approx(expected_t0)
    assert float(path.iloc[1]) == pytest.approx(expected_t1)


def test_theoretical_path_empty_score_log_returns_initial():
    path = theoretical_nav_path(pd.DataFrame(), Decimal("100"))
    assert path.iloc[0] == 100.0


# ========== AttributionReport ==========

def _build_inputs(
    initial: float = 100_000.0,
    final_nav: float = 105_000.0,
    n_ticks: int = 30,
    apy: float = 0.10,
    n_rotates: int = 2,
    gas_per_rotate: float = 10.0,
    slip_per_rotate: float = 5.0,
    lvr_per_rotate: float = 1.0,
    n_holds: int = 28,
    n_reinvests: int = 3,
):
    """构造一组一致的 nav/trade/score 日志。"""
    # NAV：从 initial 线性升到 final_nav
    nav_vals = [initial + (final_nav - initial) * i / max(n_ticks - 1, 1)
                for i in range(n_ticks)]
    nav_log = pd.DataFrame({
        "tick": range(n_ticks),
        "timestamp": [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_ticks)],
        "nav": nav_vals,
    })

    score_log = pd.DataFrame([
        {"tick": t, "pool_id": "a", "apy": apy}
        for t in range(n_ticks)
    ])

    rows = []
    for i in range(n_rotates):
        rows.append({
            "tick": i, "operation": "ROTATE",
            "from_pool_id": None, "to_pool_id": "a",
            "amount": initial,
            "gas_cost": gas_per_rotate,
            "slippage_cost": slip_per_rotate,
            "lvr_cost": lvr_per_rotate,
            "expected_gain": 0.0, "decision_reason": "OK",
        })
    for j in range(n_holds):
        rows.append({
            "tick": n_rotates + j, "operation": "HOLD",
            "from_pool_id": "a", "to_pool_id": "a",
            "amount": 0.0,
            "gas_cost": 100.0, "slippage_cost": 50.0, "lvr_cost": 10.0,  # 应被忽略
            "expected_gain": 0.0, "decision_reason": "SAME_POOL",
        })
    trade_log = pd.DataFrame(rows)

    reinvest_log = pd.DataFrame([
        {"tick": k, "pool_id": "a", "reward_compounded": 1.0,
         "gas_cost": 2.0, "expected_gain": 5.0}
        for k in range(n_reinvests)
    ])
    return nav_log, trade_log, score_log, reinvest_log


def test_attribution_basic_decomposition():
    nav, trade, score, rein = _build_inputs(
        initial=100_000, final_nav=105_000, n_ticks=30, apy=0.20,
        n_rotates=2, n_holds=28, n_reinvests=2,
        gas_per_rotate=50, slip_per_rotate=20, lvr_per_rotate=5,
    )
    rep = compute_attribution(nav, trade, score, Decimal("100000"), rein)

    assert isinstance(rep, AttributionReport)
    # 摩擦成本：2 次 rotate × 50 gas + 2 次 reinvest × 2 gas = 104
    assert rep.gas_cost == Decimal("104")
    # slippage = 2 × 20，仅 ROTATE 计入
    assert rep.slippage_cost == Decimal("40")
    assert rep.lvr_cost == Decimal("10")
    assert rep.rotation_count == 2
    assert rep.reinvest_count == 2


def test_attribution_idle_nonneg_when_actual_below_theoretical():
    nav, trade, score, rein = _build_inputs(
        initial=100_000, final_nav=101_000, n_ticks=30, apy=0.50,
        n_rotates=1, n_holds=29, n_reinvests=0,
        gas_per_rotate=10, slip_per_rotate=5, lvr_per_rotate=1,
    )
    rep = compute_attribution(nav, trade, score, Decimal("100000"), rein)
    # 高 APY + 极少摩擦 → 理论收益 >> 实际，idle 必为正
    assert rep.theoretical_total_return > rep.actual_return
    assert rep.rotation_idle_cost > 0


def test_attribution_idle_clipped_to_zero_when_actual_beats_theoretical():
    """若实际 NAV 高于理论（极端：APY 数据有负样本），idle 应被截到 0。"""
    nav, trade, score, _ = _build_inputs(
        initial=100_000, final_nav=200_000, n_ticks=30, apy=0.01,
        n_rotates=0, n_holds=30, n_reinvests=0,
    )
    rep = compute_attribution(nav, trade, score, Decimal("100000"))
    assert rep.rotation_idle_cost == Decimal(0)


def test_attribution_pct_sum_within_100_when_normal():
    """摩擦 + idle 占比加起来不超 100%（FR-08 验收口径）。"""
    nav, trade, score, rein = _build_inputs(
        initial=100_000, final_nav=104_000, n_ticks=30, apy=0.20,
        n_rotates=2, n_holds=28, n_reinvests=1,
        gas_per_rotate=50, slip_per_rotate=20, lvr_per_rotate=5,
    )
    rep = compute_attribution(nav, trade, score, Decimal("100000"), rein)
    total_pct = (rep.gas_cost_pct + rep.slippage_pct + rep.lvr_pct
                 + rep.rotation_idle_pct)
    # 摩擦 + idle 是「理论 - 实际」的全部分解，比例总和应 ≤ 100%
    assert total_pct <= Decimal(100)


def test_attribution_zero_theoretical_returns_zero_pcts():
    nav = pd.DataFrame({"nav": [100, 100, 100]})
    trade = pd.DataFrame()
    score = pd.DataFrame([{"tick": t, "pool_id": "a", "apy": 0.0} for t in range(3)])
    rep = compute_attribution(nav, trade, score, Decimal("100"))
    assert rep.gas_cost_pct == Decimal(0)
    assert rep.rotation_idle_pct == Decimal(0)


def test_attribution_to_dict_floats():
    nav, trade, score, _ = _build_inputs()
    rep = compute_attribution(nav, trade, score, Decimal("100000"))
    d = rep.to_dict()
    assert isinstance(d["theoretical_total_return"], float)
    assert isinstance(d["rotation_count"], int)


def test_attribution_export_csv(tmp_path: Path):
    nav, trade, score, _ = _build_inputs()
    rep = compute_attribution(nav, trade, score, Decimal("100000"))
    out = tmp_path / "attribution.csv"
    rep.export_csv(out)
    assert out.exists()

    df = pd.read_csv(out)
    assert len(df) == 1
    assert "theoretical_total_return" in df.columns
    assert "rotation_count" in df.columns


def test_attribution_handles_empty_nav():
    rep = compute_attribution(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        Decimal("100000"),
    )
    assert rep.theoretical_total_return == Decimal(0)
    assert rep.actual_return == Decimal(0)


def test_attribution_integrates_with_real_backtest():
    """与 BacktestEngine 实际产物联动，确保接口正确。"""
    from datetime import datetime, timedelta
    from typing import Dict, List

    from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
    from backtest.engine import BacktestEngine
    from backtest.cost_model import FrictionEstimator
    from strategy.gain_estimator import APYDeltaGainEstimator
    from strategy.interfaces import ScoringParams, WeightConfig
    from strategy.reinvest_engine import ReinvestEngine
    from strategy.rotation_engine import RotationEngine
    from strategy.scoring_engine import ScoringEngine
    from strategy.scorers.cara import CARAUtilityAdjuster
    from strategy.scorers.momentum import MomentumScorer
    from strategy.scorers.risk_penalty import (
        DownsideVolPenaltyScorer,
        MaxDrawdownPenaltyScorer,
    )

    # 简化数据：3 池 30 tick
    snapshots: List[AssetSnapshot] = []
    for t in range(30):
        pools: Dict[str, PoolMetrics] = {}
        for i, pid in enumerate(["p0", "p1", "p2"]):
            pools[pid] = PoolMetrics(
                pool_id=pid,
                apy_series=tuple(Decimal(str(0.05 + i * 0.02)) for _ in range(min(t + 1, 14))),
                tvl=Decimal("100000000"),  # 100M：让 trade/tvl 比足够小，slippage 落到 low 档
                vol_30d=Decimal("0.02"),
                token_price=Decimal("1.0"),
                gas_base_fee=Decimal("0.0000001"),
            )
        env = EnvSnapshot(
            tick=t,
            timestamp=datetime(2024, 1, 1) + timedelta(days=t),
            oracle_price={pid: Decimal("1.0") for pid in pools},
            gas_base_fee=Decimal("0.0000001"),
            gas_priority_fee=Decimal("0.00000005"),
        )
        snapshots.append(AssetSnapshot(tick=t, pools=pools, env=env))

    friction = FrictionEstimator()
    gain = APYDeltaGainEstimator()
    engine = BacktestEngine(
        initial_capital=Decimal("100000"),
        scoring_engine=ScoringEngine(
            params=ScoringParams(),
            weight_cfg=WeightConfig({
                "momentum": Decimal("0.4"), "vol_penalty": Decimal("0.25"),
                "mdd_penalty": Decimal("0.20"), "cara": Decimal("0.15"),
            }),
            scorers=[MomentumScorer(), DownsideVolPenaltyScorer(),
                     MaxDrawdownPenaltyScorer(), CARAUtilityAdjuster()],
        ),
        rotation_engine=RotationEngine(
            tau_reset=Decimal("0.01"), threshold=Decimal("0.0001"),
            gain_estimator=gain, friction_estimator=friction,
        ),
        reinvest_engine=ReinvestEngine(
            friction_estimator=friction, gain_estimator=gain,
            reinvest_window=30, risk_premium_multiplier=Decimal("1.5"),
        ),
    )
    res = engine.run(snapshots)

    rep = compute_attribution(
        res.nav_log, res.trade_log, res.score_log,
        Decimal("100000"), reinvest_log=res.reinvest_log,
    )
    # 至少 1 次轮动（首 tick 进 p2）
    assert rep.rotation_count >= 1
    # 实际收益应为正（APY 都 > 0）
    assert rep.actual_return > Decimal(0)
    # 比例守恒：actual + friction + idle = theoretical
    reconstructed = (rep.actual_return + rep.gas_cost
                     + rep.slippage_cost + rep.lvr_cost
                     + rep.rotation_idle_cost)
    assert abs(reconstructed - rep.theoretical_total_return) <= Decimal("1")
