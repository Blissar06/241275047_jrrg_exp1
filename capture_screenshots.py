"""一键截图：跑多组回测，把所有关键图表导出为 PNG 到 docs/screenshots/。

用法：
    python capture_screenshots.py

产物：
    docs/screenshots/
      ├── 01_overview_nav.png            概览 - NAV + 调仓标记 + 理论最优
      ├── 02_overview_radar_single.png   概览 - 单策略归因雷达
      ├── 03_market_apy_history.png      行情 - APY 多池历史
      ├── 04_market_tvl_history.png      行情 - TVL 多池历史
      ├── 05_market_gas_timeline.png     行情 - Gas 时序
      ├── 06_market_apy_heatmap.png      行情 - APY 池×时间热力图
      ├── 07_risk_drawdown.png           风险 - 回撤水下图
      ├── 08_risk_rolling_sharpe.png     风险 - 30 日滚动夏普
      ├── 09_position_timeline.png       仓位 - Gantt 持仓时间线
      ├── 10_cost_stacked.png            成本 - 摩擦堆积柱
      ├── 11_compare_nav_overlay.png     多策略对比 - NAV 叠加
      └── 12_compare_radar_multi.png     多策略对比 - 多边形雷达

跑 2 套数据 × 多个 preset，确保产物有代表性：
  - 合成数据 + BALANCED        → 步骤 1 演示
  - 合成数据 + 5 个预设         → 多策略对比
  - 真实链上数据 + CONSERVATIVE → 步骤 3 演示
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from backtest.cost_model import FrictionEstimator
from backtest.engine import BacktestEngine, BacktestResult
from data.sample_data import generate_sample_data
from data_model.loader import build_asset_snapshots, load_gas_csv, load_pool_csv
from report.attribution import (
    AttributionReport,
    compute_attribution,
    theoretical_nav_path,
)
from report.metrics import MetricsReport, compute_metrics
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import ScoringParams, WeightConfig
from strategy.presets import (
    AGGRESSIVE_MOMENTUM,
    ALL_PRESETS,
    BALANCED,
    CONSERVATIVE,
    EXTREME_RISK_AVERSE,
    LOW_FREQUENCY_VALUE,
    StrategyPreset,
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
from ui import charts


# =================================================================
# 输出目录
# =================================================================

OUT_DIR = PROJECT_ROOT / "docs" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLIPPAGE_PRESETS = {
    "低（流动性充足）": (Decimal("0.0005"), Decimal("0.002"), Decimal("0.005")),
    "中（默认）":     (Decimal("0.001"),  Decimal("0.003"), Decimal("0.008")),
    "高（流动性紧张）": (Decimal("0.002"),  Decimal("0.006"), Decimal("0.015")),
}


def _build_engine(preset: StrategyPreset) -> BacktestEngine:
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
    )


def _run(preset, snapshots) -> dict:
    engine = _build_engine(preset)
    result = engine.run(snapshots)
    m = compute_metrics(result.nav_log, result.trade_log, result.reinvest_log)
    a = compute_attribution(
        result.nav_log, result.trade_log, result.score_log,
        Decimal("100000"), reinvest_log=result.reinvest_log,
    )
    theo = theoretical_nav_path(result.score_log, Decimal("100000"))
    return {"preset": preset, "result": result, "metrics": m, "attribution": a, "theo": theo}


# =================================================================
# 保存工具
# =================================================================

def save(fig, name: str, *, width: int = 1280, height: int = 600) -> None:
    out = OUT_DIR / name
    # 加白底防止 streamlit dark theme 默认透明背景
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Microsoft YaHei, Arial", size=14, color="#222"),
    )
    fig.write_image(str(out), width=width, height=height, scale=2)
    print(f"  [saved] {out.name}  ({out.stat().st_size // 1024} KB)")


# =================================================================
# 主流程
# =================================================================

def main():
    print("=" * 64)
    print("第 1 组：合成数据 + BALANCED 预设  → 概览 / 行情 / 风险 / 仓位 / 成本")
    print("=" * 64)

    pool_df_syn, gas_df_syn = generate_sample_data(n_days=365)
    snaps_syn = build_asset_snapshots(
        pool_df_syn, gas_df_syn, config={"momentum_window": 14},
    )

    run_bal = _run(BALANCED, snaps_syn)
    res, m, a, theo = run_bal["result"], run_bal["metrics"], run_bal["attribution"], run_bal["theo"]
    print(f"  BALANCED: 年化={float(m.annualized_return)*100:.2f}%  "
          f"MDD={float(m.max_drawdown)*100:.2f}%  Sharpe={float(m.sharpe_ratio):.3f}  "
          f"调仓={a.rotation_count}")

    # 01 概览 NAV
    save(
        charts.nav_with_trade_markers(res.nav_log, res.trade_log, theoretical_nav=theo),
        "01_overview_nav.png",
    )
    # 02 概览雷达（单策略）
    save(
        charts.attribution_radar_multi({"BALANCED": a}),
        "02_overview_radar_single.png", width=900, height=600,
    )
    # 03 行情 APY
    save(charts.apy_history(pool_df_syn), "03_market_apy_history.png")
    # 04 行情 TVL
    save(charts.tvl_history(pool_df_syn), "04_market_tvl_history.png")
    # 05 行情 Gas
    save(
        charts.gas_timeline(gas_df_syn, nav_log=res.nav_log),
        "05_market_gas_timeline.png",
    )
    # 06 APY 热力图
    save(
        charts.apy_heatmap(pool_df_syn),
        "06_market_apy_heatmap.png", width=1280, height=400,
    )

    # ----- 风险 / 仓位 / 成本 用真实链上数据更醒目 -----

    print()
    print("=" * 64)
    print("第 2 组：真实链上数据 + CONSERVATIVE 预设  → 风险 / 仓位 / 成本")
    print("=" * 64)

    real_pool = Path("data/real_pools.csv")
    real_gas = Path("data/real_gas.csv")
    if not (real_pool.exists() and real_gas.exists()):
        print(f"  [skip] 真实数据不存在；用合成数据替代")
        run_cons = _run(CONSERVATIVE, snaps_syn)
    else:
        pool_df_real = load_pool_csv(real_pool)
        gas_df_real = load_gas_csv(real_gas)
        snaps_real = build_asset_snapshots(
            pool_df_real, gas_df_real, config={"momentum_window": 14},
        )
        run_cons = _run(CONSERVATIVE, snaps_real)

    res_c, m_c, a_c = run_cons["result"], run_cons["metrics"], run_cons["attribution"]
    print(f"  CONSERVATIVE: 年化={float(m_c.annualized_return)*100:.2f}%  "
          f"MDD={float(m_c.max_drawdown)*100:.2f}%  Sharpe={float(m_c.sharpe_ratio):.3f}  "
          f"调仓={a_c.rotation_count}")

    # 07 回撤
    save(
        charts.drawdown_underwater(res_c.nav_log),
        "07_risk_drawdown.png", width=1280, height=400,
    )
    # 08 滚动夏普
    save(
        charts.rolling_sharpe(res_c.nav_log, window=30),
        "08_risk_rolling_sharpe.png", width=1280, height=400,
    )
    # 09 持仓时间线
    save(
        charts.position_timeline(res_c.nav_log),
        "09_position_timeline.png", width=1280, height=300,
    )
    # 10 成本堆积
    save(
        charts.cost_composition_stacked(res_c.trade_log),
        "10_cost_stacked.png", width=1280, height=400,
    )

    # ----- 多策略对比 用合成数据 -----

    print()
    print("=" * 64)
    print("第 3 组：合成数据 × 5 预设  → 多策略对比")
    print("=" * 64)

    runs = {}
    for p in ALL_PRESETS:
        r = _run(p, snaps_syn)
        runs[p.name] = r
        print(f"  {p.name:14s}: 年化={float(r['metrics'].annualized_return)*100:>6.2f}%  "
              f"MDD={float(r['metrics'].max_drawdown)*100:>5.2f}%  "
              f"Sharpe={float(r['metrics'].sharpe_ratio):>7.3f}  "
              f"调仓={r['attribution'].rotation_count}")

    # 11 NAV 叠加
    import plotly.graph_objects as go
    fig_nav = go.Figure()
    for i, (name, run) in enumerate(runs.items()):
        fig_nav.add_trace(go.Scatter(
            x=run["result"].nav_log["timestamp"],
            y=run["result"].nav_log["nav"],
            mode="lines", name=name,
            line=dict(color=charts._color_for(i), width=2),
        ))
    fig_nav.update_layout(
        height=520,
        xaxis_title="时间", yaxis_title="NAV",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    save(fig_nav, "11_compare_nav_overlay.png", width=1280, height=520)

    # 12 多边形雷达
    save(
        charts.attribution_radar_multi({n: r["attribution"] for n, r in runs.items()}),
        "12_compare_radar_multi.png", width=900, height=700,
    )

    # ----- 摘要表 -----

    print()
    print("=" * 64)
    print("汇总产物:")
    print("=" * 64)
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.relative_to(PROJECT_ROOT)}  ({f.stat().st_size // 1024} KB)")
    print(f"\n共 {len(list(OUT_DIR.glob('*.png')))} 张 PNG 已落到 {OUT_DIR}")


if __name__ == "__main__":
    main()
