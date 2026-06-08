"""Phase 3 命令 3-3：BacktestEngine 集成测试。

覆盖：
  - 主流程跑通：snapshots 全部处理、四张 log 生成
  - 复现性（NFR-02）：相同输入两次 run() 输出 NAV 序列一致
  - DataIntegrityError 引擎不崩溃（fallback HOLD）
  - 压力事件正确传播：Gas_Spike 后 trade_log 该 tick 的 gas_cost 翻倍
  - Parquet 落盘：4 个文件被写出，列名正确
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from backtest.cost_model import FrictionEstimator
from backtest.engine import BacktestEngine, BacktestResult
from backtest.event_injector import EventInjector, EventType, StressEvent
from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import ScoringParams, WeightConfig
from strategy.reinvest_engine import ReinvestEngine
from strategy.rotation_engine import RotationEngine
from strategy.scorers.cara import CARAUtilityAdjuster
from strategy.scorers.momentum import MomentumScorer
from strategy.scorers.risk_penalty import (
    DownsideVolPenaltyScorer,
    MaxDrawdownPenaltyScorer,
)
from strategy.scoring_engine import ScoringEngine


# ============================================================
# 数据生成
# ============================================================

def _make_snapshots(
    n_ticks: int = 30,
    n_pools: int = 3,
    apy_seed: float = 0.05,
    apy_step: float = 0.01,
    base_fee: str = "0.0000001",
) -> List[AssetSnapshot]:
    """生成 n_ticks 个 snapshots，n_pools 个池，APY 平稳上升。"""
    snapshots: List[AssetSnapshot] = []
    pool_ids = [f"p{i}" for i in range(n_pools)]
    start = datetime(2024, 1, 1)
    for t in range(n_ticks):
        pools: Dict[str, PoolMetrics] = {}
        for i, pid in enumerate(pool_ids):
            apy = apy_seed + i * apy_step
            history = tuple(Decimal(str(apy + 0.0001 * k)) for k in range(min(t + 1, 14)))
            pools[pid] = PoolMetrics(
                pool_id=pid,
                apy_series=history,
                tvl=Decimal("1000000"),
                vol_30d=Decimal("0.02"),
                token_price=Decimal("1.0"),
                gas_base_fee=Decimal(base_fee),
            )
        env = EnvSnapshot(
            tick=t,
            timestamp=start + timedelta(days=t),
            oracle_price={pid: Decimal("1.0") for pid in pool_ids},
            gas_base_fee=Decimal(base_fee),
            gas_priority_fee=Decimal("0.00000005"),
        )
        snapshots.append(AssetSnapshot(tick=t, pools=pools, env=env))
    return snapshots


def _build_engine(
    initial_capital: str = "100000",
    threshold: str = "0.001",
    tau: str = "0.01",
    event_injector: EventInjector | None = None,
) -> BacktestEngine:
    scoring = ScoringEngine(
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
    friction = FrictionEstimator()
    gain = APYDeltaGainEstimator()
    rotation = RotationEngine(
        tau_reset=Decimal(tau),
        threshold=Decimal(threshold),
        gain_estimator=gain,
        friction_estimator=friction,
        gain_horizon_ticks=30,
    )
    reinvest = ReinvestEngine(
        friction_estimator=friction,
        gain_estimator=gain,
        reinvest_window=30,
        risk_premium_multiplier=Decimal("1.5"),
    )
    return BacktestEngine(
        initial_capital=Decimal(initial_capital),
        scoring_engine=scoring,
        rotation_engine=rotation,
        reinvest_engine=reinvest,
        event_injector=event_injector,
    )


# ============================================================
# 基本流程
# ============================================================

def test_run_processes_all_snapshots():
    engine = _build_engine()
    snaps = _make_snapshots(n_ticks=30, n_pools=3)
    res = engine.run(snaps)

    assert isinstance(res, BacktestResult)
    assert res.snapshots_processed == 30
    assert len(res.nav_log) == 30
    # 每 tick × 3 池 → score_log 应有 90 行
    assert len(res.score_log) == 90
    # trade_log 每 tick 至少 1 条（ROTATE 或 HOLD）
    assert len(res.trade_log) == 30


def test_initial_position_invests_into_top_pool():
    """首 tick 应开仓到 top-1（APY 最高的 p2）。"""
    engine = _build_engine(threshold="0.0001")
    snaps = _make_snapshots(n_ticks=2, n_pools=3, apy_step=0.05)
    res = engine.run(snaps)

    first_trade = res.trade_log.iloc[0]
    assert first_trade["operation"] == "ROTATE"
    assert first_trade["to_pool_id"] == "p2"  # APY 最高


def test_nav_grows_when_apy_positive():
    """APY 持续为正时 NAV 应单调或近单调上升。"""
    engine = _build_engine(threshold="0.0001")
    snaps = _make_snapshots(n_ticks=60, n_pools=3, apy_seed=0.10, apy_step=0.01)
    res = engine.run(snaps)
    nav_series = res.nav_log["nav"].values
    assert nav_series[-1] > nav_series[0]


def test_score_log_columns_include_all_components():
    engine = _build_engine()
    snaps = _make_snapshots(n_ticks=5, n_pools=3)
    res = engine.run(snaps)
    cols = set(res.score_log.columns)
    assert {"tick", "timestamp", "pool_id", "total_score",
            "momentum", "vol_penalty", "mdd_penalty", "cara"} <= cols


def test_trade_log_includes_hold_records_with_reason():
    """高 threshold 必然 GATE_FAIL，HOLD 记录应携带 GATE_FAIL reason。"""
    engine = _build_engine(threshold="100")  # 不可能的高门槛
    snaps = _make_snapshots(n_ticks=5, n_pools=3)
    res = engine.run(snaps)
    holds = res.trade_log[res.trade_log["operation"] == "HOLD"]
    assert len(holds) == 5
    # 至少一些 HOLD 是 GATE_FAIL（首 tick 也可能被门槛挡住）
    reasons = set(holds["decision_reason"])
    assert "GATE_FAIL" in reasons


# ============================================================
# 复现性（NFR-02）
# ============================================================

def test_two_runs_produce_identical_nav_series():
    snaps = _make_snapshots(n_ticks=40, n_pools=4)
    e1 = _build_engine()
    e2 = _build_engine()
    r1 = e1.run(snaps)
    r2 = e2.run(snaps)

    assert np.allclose(r1.nav_log["nav"].values, r2.nav_log["nav"].values, rtol=0, atol=0)
    # DataFrame.equals 处理 None/NaN 同位置的相等；DataFrame == 比较 None 会返回 False
    assert r1.trade_log.equals(r2.trade_log)
    assert r1.score_log.equals(r2.score_log)


# ============================================================
# 数据完整性 fallback (E-RT-001)
# ============================================================

def test_engine_survives_empty_apy_series_in_one_tick():
    snaps = _make_snapshots(n_ticks=10, n_pools=3)
    # 把第 5 个 tick 的 p1 改成空 apy_series
    bad_pool = PoolMetrics(
        pool_id="p1",
        apy_series=(),
        tvl=Decimal("1000000"),
        vol_30d=Decimal("0.02"),
        token_price=Decimal("1.0"),
        gas_base_fee=Decimal("0.0000001"),
    )
    pools_patched = dict(snaps[5].pools)
    pools_patched["p1"] = bad_pool
    snaps[5] = AssetSnapshot(tick=5, pools=pools_patched, env=snaps[5].env)

    engine = _build_engine()
    res = engine.run(snaps)
    # 引擎必须跑完所有 tick，不应抛异常
    assert res.snapshots_processed == 10


# ============================================================
# 压力事件传播
# ============================================================

def test_gas_spike_event_inflates_env_in_target_window():
    """GAS_SPIKE 应该让事件窗口内的 env.gas_base_fee 暴涨；通过 nav_log 直接观测。"""
    inj = EventInjector([
        StressEvent(EventType.GAS_SPIKE, start_tick=5, duration=1, impact_ratio=Decimal("4.0")),
    ])
    engine = _build_engine(threshold="0.0001", event_injector=inj)
    snaps = _make_snapshots(n_ticks=10, n_pools=3, apy_step=0.05)
    res = engine.run(snaps)

    base_4 = res.nav_log.loc[res.nav_log["tick"] == 4, "env_gas_base_fee"].iloc[0]
    base_5 = res.nav_log.loc[res.nav_log["tick"] == 5, "env_gas_base_fee"].iloc[0]
    base_6 = res.nav_log.loc[res.nav_log["tick"] == 6, "env_gas_base_fee"].iloc[0]
    # 窗口内放大 5 倍（1 + 4.0），窗口外恢复
    assert base_5 == pytest.approx(base_4 * 5)
    assert base_6 == pytest.approx(base_4)


def test_pool_exploit_event_drops_pool_score():
    """目标池 APY 暴跌后，下一次评分中应排到末位。"""
    inj = EventInjector([
        StressEvent(
            EventType.POOL_EXPLOIT, start_tick=5, duration=1,
            impact_ratio=Decimal("0.95"), target_pool_id="p2",
        ),
    ])
    engine = _build_engine(event_injector=inj)
    snaps = _make_snapshots(n_ticks=10, n_pools=3, apy_step=0.02)
    res = engine.run(snaps)

    # tick=5 时 p2 的 total_score 应低于 tick=4
    score_p2_t4 = res.score_log.query("tick==4 and pool_id=='p2'")["total_score"].iloc[0]
    score_p2_t5 = res.score_log.query("tick==5 and pool_id=='p2'")["total_score"].iloc[0]
    assert score_p2_t5 < score_p2_t4


# ============================================================
# Parquet 落盘
# ============================================================

def test_persist_creates_four_parquet_files(tmp_path: Path):
    engine = _build_engine()
    snaps = _make_snapshots(n_ticks=10, n_pools=3)
    out = tmp_path / "logs"
    engine.run(snaps, output_dir=out)

    assert (out / "nav_log.parquet").exists()
    assert (out / "trade_log.parquet").exists()
    assert (out / "reinvest_log.parquet").exists()
    assert (out / "score_log.parquet").exists()


def test_persisted_logs_round_trip(tmp_path: Path):
    import pandas as pd
    engine = _build_engine()
    snaps = _make_snapshots(n_ticks=10, n_pools=3)
    out = tmp_path / "logs"
    res = engine.run(snaps, output_dir=out)

    nav_back = pd.read_parquet(out / "nav_log.parquet")
    assert len(nav_back) == len(res.nav_log)
    assert set(nav_back.columns) == set(res.nav_log.columns)


# ============================================================
# 性能（粗测：1000 tick × 3 池应秒级完成）
# ============================================================

def test_perf_smoke_1000_ticks_under_5_seconds():
    import time
    engine = _build_engine()
    snaps = _make_snapshots(n_ticks=1000, n_pools=5)
    t0 = time.perf_counter()
    res = engine.run(snaps)
    dt = time.perf_counter() - t0
    assert res.snapshots_processed == 1000
    # 1000 tick 5 池作为线性外推下 spec 的 10000×10 场景的 1/20
    # 在合理硬件上应 < 5 秒
    assert dt < 5.0, f"perf regression: {dt:.2f}s for 1000 ticks × 5 pools"
