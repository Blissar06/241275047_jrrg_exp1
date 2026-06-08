"""端到端示例脚本：CSV → 加载 → 回测 → 报表 + 验收断言。

用法：
    python run_example.py
    python run_example.py --regen   # 强制重新生成 CSV
    python run_example.py --no-events  # 跳过压力事件

执行步骤：
  1. 在 data/ 下生成 pools_sample.csv / gas_sample.csv（若不存在或 --regen）
  2. 通过 load_pool_csv / load_gas_csv 重新加载（验证 CSV 闭环）
  3. 构建 AssetSnapshot 列表（365 个 tick，3 个池）
  4. 注入压力事件：Gas_Spike (tick 150~154, 5×) + Pool_Exploit (Curve_3Pool, tick 200, -90%)
  5. 跑 BacktestEngine.run()
  6. 打印 MetricsReport / AttributionReport
  7. 落盘 Parquet（data/output/）+ CSV（data/output/）
  8. 验收断言：见末尾 _verify()
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from decimal import Decimal
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.cost_model import FrictionEstimator
from backtest.engine import BacktestEngine, BacktestResult
from backtest.event_injector import EventInjector, EventType, StressEvent
from data.sample_data import generate_sample_data
from data_model.loader import (
    build_asset_snapshots,
    load_gas_csv,
    load_pool_csv,
)
from report.attribution import AttributionReport, compute_attribution
from report.metrics import MetricsReport, compute_metrics
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


POOL_IDS = ["Aave_USDC", "Curve_3Pool", "Convex_ETH"]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"

POOL_CSV = DATA_DIR / "pools_sample.csv"
GAS_CSV = DATA_DIR / "gas_sample.csv"


# =================================================================
# CSV 写入
# =================================================================

def ensure_sample_csvs(force: bool = False) -> None:
    """生成示例 CSV（若不存在或 force）。"""
    if POOL_CSV.exists() and GAS_CSV.exists() and not force:
        print(f"[skip] {POOL_CSV.name} 与 {GAS_CSV.name} 已存在；用 --regen 强制重新生成")
        return

    pool_df, gas_df = generate_sample_data(
        pool_ids=POOL_IDS,
        crash_pool_index=1,   # Curve_3Pool 闪崩
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool_df.to_csv(POOL_CSV, index=False)
    gas_df.to_csv(GAS_CSV, index=False)
    print(f"[gen] {POOL_CSV} ({len(pool_df)} rows)")
    print(f"[gen] {GAS_CSV} ({len(gas_df)} rows)")


# =================================================================
# 引擎构建
# =================================================================

def build_engine(events: list[StressEvent] | None) -> BacktestEngine:
    friction = FrictionEstimator()
    gain = APYDeltaGainEstimator()
    return BacktestEngine(
        initial_capital=Decimal("100000"),
        scoring_engine=ScoringEngine(
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
        ),
        rotation_engine=RotationEngine(
            tau_reset=Decimal("0.05"),
            threshold=Decimal("0.001"),
            gain_estimator=gain,
            friction_estimator=friction,
            gain_horizon_ticks=30,
        ),
        reinvest_engine=ReinvestEngine(
            friction_estimator=friction,
            gain_estimator=gain,
            reinvest_window=30,
            risk_premium_multiplier=Decimal("1.5"),
        ),
        event_injector=EventInjector(events) if events else None,
    )


def default_events() -> list[StressEvent]:
    """事件计划。

    Gas_Spike 已经写在 CSV 里（CSV 是「历史」），EventInjector 仅做「what-if」叠加，
    这里只演示 Pool_Exploit；同时叠加 Gas_Spike 会与 CSV 双倍生效（ratio=25× 而非 5×）。
    """
    return [
        StressEvent(
            EventType.POOL_EXPLOIT, start_tick=200, duration=1,
            impact_ratio=Decimal("0.9"),    # APY × 0.1
            target_pool_id="Curve_3Pool",
        ),
    ]


# =================================================================
# 主流程
# =================================================================

def run(events_enabled: bool = True, regen: bool = False) -> tuple[
    BacktestResult, MetricsReport, AttributionReport, float
]:
    ensure_sample_csvs(force=regen)

    pool_df = load_pool_csv(POOL_CSV)
    gas_df = load_gas_csv(GAS_CSV)
    snapshots = build_asset_snapshots(
        pool_df, gas_df, config={"momentum_window": 14},
    )
    print(f"[load] {len(snapshots)} snapshots from CSV")

    events = default_events() if events_enabled else None
    engine = build_engine(events)

    t0 = time.perf_counter()
    result = engine.run(snapshots)
    dt = time.perf_counter() - t0
    print(f"[run] {result.snapshots_processed} ticks in {dt:.2f}s")

    metrics = compute_metrics(result.nav_log, result.trade_log, result.reinvest_log)
    attribution = compute_attribution(
        result.nav_log, result.trade_log, result.score_log,
        Decimal("100000"), reinvest_log=result.reinvest_log,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.persist(OUTPUT_DIR)
    pd.DataFrame([metrics.to_dict()]).to_csv(
        OUTPUT_DIR / "metrics_report.csv", index=False,
    )
    attribution.export_csv(OUTPUT_DIR / "attribution_report.csv")
    print(f"[save] logs → {OUTPUT_DIR}")

    return result, metrics, attribution, dt


# =================================================================
# 报表打印
# =================================================================

def print_metrics(metrics: MetricsReport) -> None:
    print("\n=== 绩效指标 ===")
    print(f"  N periods           : {metrics.n_periods}")
    print(f"  Annualized return   : {float(metrics.annualized_return) * 100:>8.2f} %")
    print(f"  Annualized vol      : {float(metrics.annualized_volatility) * 100:>8.2f} %")
    print(f"  Max drawdown        : {float(metrics.max_drawdown) * 100:>8.2f} %")
    print(f"  Sharpe ratio        : {float(metrics.sharpe_ratio):>8.3f}")
    print(f"  Sortino ratio       : {float(metrics.sortino_ratio):>8.3f}")
    print(f"  Calmar ratio        : {float(metrics.calmar_ratio):>8.3f}")
    print(f"  Total gas cost      : {float(metrics.total_gas_cost):>10.2f}")
    print(f"  Total slippage cost : {float(metrics.total_slippage_cost):>10.2f}")
    print(f"  Total LVR cost      : {float(metrics.total_lvr_cost):>10.2f}")
    print(f"  Total friction cost : {float(metrics.total_friction_cost):>10.2f}")


def print_attribution(attr: AttributionReport) -> None:
    print("\n=== 收益归因 ===")
    print(f"  Theoretical return  : {float(attr.theoretical_total_return):>10.2f}")
    print(f"  Actual return       : {float(attr.actual_return):>10.2f}")
    print(f"  Rotations           : {attr.rotation_count}")
    print(f"  Reinvests           : {attr.reinvest_count}")
    print(f"  Gas pct             : {float(attr.gas_cost_pct):>8.2f} %")
    print(f"  Slippage pct        : {float(attr.slippage_pct):>8.2f} %")
    print(f"  LVR pct             : {float(attr.lvr_pct):>8.2f} %")
    print(f"  Rotation idle pct   : {float(attr.rotation_idle_pct):>8.2f} %")
    print(f"  Total friction pct  : {float(attr.total_friction_pct):>8.2f} %")


# =================================================================
# 验收断言
# =================================================================

def verify(
    result: BacktestResult,
    metrics: MetricsReport,
    attribution: AttributionReport,
    elapsed_seconds: float,
) -> None:
    print("\n=== 验收检查 ===")
    issues: list[str] = []

    def _ok(label: str, condition: bool, detail: str = "") -> None:
        marker = "PASS" if condition else "FAIL"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{marker}] {label}{suffix}")
        if not condition:
            issues.append(label)

    _ok("snapshots_processed == 365", result.snapshots_processed == 365,
        f"got {result.snapshots_processed}")
    _ok("runtime < 5s", elapsed_seconds < 5.0, f"{elapsed_seconds:.2f}s")
    _ok("nav_log len == 365", len(result.nav_log) == 365)

    # CSV 中 Gas_Spike (tick 150~154) → env.gas_base_fee 翻 5×
    base_fee_outside = float(
        result.nav_log.loc[result.nav_log["tick"] == 100, "env_gas_base_fee"].iloc[0]
    )
    base_fee_inside = float(
        result.nav_log.loc[result.nav_log["tick"] == 152, "env_gas_base_fee"].iloc[0]
    )
    _ok("CSV Gas_Spike 期间 env_gas_base_fee 翻 5×",
        abs(base_fee_inside / base_fee_outside - 5.0) < 0.01,
        f"ratio={base_fee_inside / base_fee_outside:.2f}")

    # Pool_Exploit 注入后，被攻击池的 score 应明显下跌
    score_t199 = result.score_log[
        (result.score_log["tick"] == 199) & (result.score_log["pool_id"] == "Curve_3Pool")
    ]["total_score"].iloc[0]
    score_t200 = result.score_log[
        (result.score_log["tick"] == 200) & (result.score_log["pool_id"] == "Curve_3Pool")
    ]["total_score"].iloc[0]
    _ok("Pool_Exploit 后 Curve_3Pool 评分下跌",
        score_t200 < score_t199,
        f"score 199→200: {score_t199:.3f} → {score_t200:.3f}")

    # tick 200 触发 Pool_Exploit：若策略持有 Curve_3Pool 应触发轮动；否则不算失败
    rotates = result.trade_log[result.trade_log["operation"] == "ROTATE"]
    rotate_around_200 = rotates[(rotates["tick"] >= 199) & (rotates["tick"] <= 205)]
    if len(rotate_around_200) > 0:
        print(f"  [INFO] tick 200 附近触发 {len(rotate_around_200)} 次轮动（策略持有受灾池时）")
    else:
        print("  [INFO] tick 200 附近未触发 ROTATE（策略未持有 Curve_3Pool 时正常）")

    _ok("total_gas_cost > 0", metrics.total_gas_cost > Decimal(0))
    _ok("total_lvr_cost > 0", metrics.total_lvr_cost > Decimal(0),
        f"{float(metrics.total_lvr_cost):.4f}")
    _ok("attribution.actual_return ≠ 0", attribution.actual_return != Decimal(0))

    if issues:
        print(f"\n!! 验收失败项: {issues}")
        sys.exit(1)
    print("\n=== 全部验收通过 ===")


# =================================================================
# CLI
# =================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeFi 回测端到端示例")
    parser.add_argument("--regen", action="store_true", help="强制重新生成 CSV")
    parser.add_argument("--no-events", action="store_true", help="跳过压力事件注入")
    parser.add_argument("--no-verify", action="store_true", help="跳过验收断言（只跑流程）")
    parser.add_argument("--quiet", action="store_true", help="抑制 INFO 日志")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    result, metrics, attribution, dt = run(
        events_enabled=not args.no_events,
        regen=args.regen,
    )
    print_metrics(metrics)
    print_attribution(attribution)
    if not args.no_verify:
        verify(result, metrics, attribution, dt)


if __name__ == "__main__":
    main()
