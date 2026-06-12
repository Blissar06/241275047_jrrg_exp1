"""Streamlit 交互看板（FR-09 完整版）。

启动方式：
    streamlit run ui/app.py

完整功能：
  - 数据源切换：合成数据 / 真实链上数据（DefiLlama，需先用 fetch_data.py 下载）
  - 多 Tab 主区域：概览 / 行情 / 风险 / 仓位 / 成本 / 多策略对比
  - 9 张图表（NAV+调仓、APY/TVL/Gas 三类时序、热力图、Drawdown、Rolling Sharpe、
    Position Gantt、成本堆积、归因雷达）
  - 多策略 A/B/C 同屏对比（保存到 session_state，叠加 NAV + 多边形雷达 + 指标表）
  - CSV 下载（归因 / 交易 / NAV）
"""
from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

# 让 streamlit run ui/app.py 在任何 cwd 都能 import 本项目
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from backtest.cost_model import FrictionEstimator
from backtest.engine import BacktestEngine, BacktestResult
from backtest.event_injector import EventInjector, EventType, StressEvent
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
from strategy.reinvest_engine import ReinvestEngine
from strategy.rotation_engine import RotationEngine
from strategy.presets import (
    ALL_PRESETS,
    StrategyPreset,
    get_preset,
    list_preset_names,
)
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
# 配置常量
# =================================================================

SLIPPAGE_PRESETS = {
    "低（流动性充足）": (Decimal("0.0005"), Decimal("0.002"), Decimal("0.005")),
    "中（默认）":     (Decimal("0.001"),  Decimal("0.003"), Decimal("0.008")),
    "高（流动性紧张）": (Decimal("0.002"),  Decimal("0.006"), Decimal("0.015")),
}

EVENT_PRESETS = {
    "Gas_Spike (tick 150~154, 5×)": StressEvent(
        EventType.GAS_SPIKE, start_tick=150, duration=5, impact_ratio=Decimal("4.0"),
    ),
    "Pool_Exploit (pool_B, tick 200, -90%)": StressEvent(
        EventType.POOL_EXPLOIT, start_tick=200, duration=1,
        impact_ratio=Decimal("0.9"), target_pool_id="pool_B",
    ),
    "Liquidity_Dryup (pool_C, tick 250~259, -80%)": StressEvent(
        EventType.LIQUIDITY_DRYUP, start_tick=250, duration=10,
        impact_ratio=Decimal("0.8"), target_pool_id="pool_C",
    ),
}


# =================================================================
# 数据加载
# =================================================================

@st.cache_data
def _load_synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_sample_data()


@st.cache_data
def _load_real_csvs(
    pool_path: str, gas_path: str,
    _pool_mtime: float, _gas_mtime: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """带 mtime 的缓存键 —— 文件变了自动重读，不需要手动 clear cache。

    `_pool_mtime` / `_gas_mtime` 以下划线开头：streamlit 仍把它们计入缓存键，
    但不会作为函数参数透传（约定俗成）。
    """
    return load_pool_csv(pool_path), load_gas_csv(gas_path)


def _load_real_csvs_auto(pool_path: str, gas_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """调用方便函数：自动取 mtime 喂给 _load_real_csvs。"""
    p_mtime = Path(pool_path).stat().st_mtime
    g_mtime = Path(gas_path).stat().st_mtime
    return _load_real_csvs(pool_path, gas_path, p_mtime, g_mtime)


# =================================================================
# 引擎构建 + 回测运行
# =================================================================

def build_engine(
    initial_capital: Decimal,
    threshold: Decimal,
    tau_reset: Decimal,
    cara_alpha: Decimal,
    weights: dict[str, Decimal],
    slip_low: Decimal,
    slip_mid: Decimal,
    slip_high: Decimal,
    event_injector: EventInjector | None,
    gain_horizon_ticks: int = 30,
) -> BacktestEngine:
    scoring = ScoringEngine(
        params=ScoringParams(cara_alpha=cara_alpha),
        weight_cfg=WeightConfig(weights),
        scorers=[
            MomentumScorer(),
            DownsideVolPenaltyScorer(),
            MaxDrawdownPenaltyScorer(),
            CARAUtilityAdjuster(),
            TokenPriceVolPenaltyScorer(),
            TokenPriceMDDPenaltyScorer(),
        ],
    )
    friction = FrictionEstimator(
        slip_rate_low=slip_low, slip_rate_mid=slip_mid, slip_rate_high=slip_high,
    )
    gain = APYDeltaGainEstimator()
    rotation = RotationEngine(
        tau_reset=tau_reset, threshold=threshold,
        gain_estimator=gain, friction_estimator=friction,
        gain_horizon_ticks=gain_horizon_ticks,
    )
    reinvest = ReinvestEngine(
        friction_estimator=friction, gain_estimator=gain,
        reinvest_window=30, risk_premium_multiplier=Decimal("1.5"),
    )
    return BacktestEngine(
        initial_capital=initial_capital,
        scoring_engine=scoring,
        rotation_engine=rotation,
        reinvest_engine=reinvest,
        event_injector=event_injector,
    )


def run_backtest(params: dict[str, Any]) -> dict[str, Any]:
    """根据 params 完整跑一次回测，返回打包后的产物（含 pool_df 用于行情图）。"""
    if params["data_source"] == "synthetic":
        pool_df, gas_df = _load_synthetic()
    else:
        pool_df, gas_df = _load_real_csvs_auto(params["pool_csv"], params["gas_csv"])

    # 按用户选择的日期范围截取（含端点）
    date_from = params.get("date_from")
    date_to = params.get("date_to")
    if date_from is not None and date_to is not None:
        df_from = pd.Timestamp(date_from)
        df_to = pd.Timestamp(date_to)
        pool_df = pool_df[(pool_df["timestamp"] >= df_from)
                          & (pool_df["timestamp"] <= df_to)].reset_index(drop=True)
        gas_df = gas_df[(gas_df["timestamp"] >= df_from)
                        & (gas_df["timestamp"] <= df_to)].reset_index(drop=True)
        if pool_df.empty or gas_df.empty:
            raise ValueError(
                f"选择的日期范围 [{date_from} ~ {date_to}] 内没有数据"
            )

    snapshots = build_asset_snapshots(
        pool_df, gas_df, config={"momentum_window": 14},
    )

    event_list = [EVENT_PRESETS[name] for name in params["selected_events"]]
    injector = EventInjector(event_list) if event_list else None

    slip_low, slip_mid, slip_high = SLIPPAGE_PRESETS[params["slip_choice"]]
    engine = build_engine(
        initial_capital=Decimal(str(params["initial_capital"])),
        threshold=Decimal(str(params["threshold"])),
        tau_reset=Decimal(str(params["tau_reset"])),
        cara_alpha=Decimal(str(params["cara_alpha"])),
        weights={
            "momentum": Decimal(str(params["w_mom"])),
            "vol_penalty": Decimal(str(params["w_vol"])),
            "mdd_penalty": Decimal(str(params["w_mdd"])),
            "cara": Decimal(str(params["w_cara"])),
            "price_vol_penalty": Decimal(str(params["w_pvol"])),
            "price_mdd_penalty": Decimal(str(params["w_pmdd"])),
        },
        slip_low=slip_low, slip_mid=slip_mid, slip_high=slip_high,
        event_injector=injector,
        gain_horizon_ticks=int(params["gain_horizon"]),
    )
    result = engine.run(snapshots)
    metrics = compute_metrics(result.nav_log, result.trade_log, result.reinvest_log)
    attribution = compute_attribution(
        result.nav_log, result.trade_log, result.score_log,
        Decimal(str(params["initial_capital"])),
        reinvest_log=result.reinvest_log,
    )
    theo = theoretical_nav_path(
        result.score_log, Decimal(str(params["initial_capital"])),
    )
    return {
        "params": params,
        "pool_df": pool_df,
        "gas_df": gas_df,
        "result": result,
        "metrics": metrics,
        "attribution": attribution,
        "theoretical_nav": theo,
    }


# =================================================================
# 渲染辅助
# =================================================================

def render_metric_cards(metrics: MetricsReport, attribution: AttributionReport) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年化收益", f"{float(metrics.annualized_return) * 100:.2f}%")
    c2.metric("夏普比率", f"{float(metrics.sharpe_ratio):.3f}")
    c3.metric("索提诺比率", f"{float(metrics.sortino_ratio):.3f}")
    c4.metric("最大回撤", f"{float(metrics.max_drawdown) * 100:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("调仓次数", attribution.rotation_count)
    c6.metric("复投次数", attribution.reinvest_count)
    total_fr = float(metrics.total_friction_cost)
    initial_for_pct = float(attribution.actual_return) + total_fr  # ≈ initial capital + return
    fr_pct = total_fr / max(initial_for_pct, 1.0) * 100
    c7.metric("总摩擦成本", f"{total_fr:.2f}", delta=f"{fr_pct:.2f}% / 总资本")
    c8.metric("调仓空窗折损", f"{float(attribution.rotation_idle_cost):.2f}",
              delta=f"{float(attribution.rotation_idle_pct):.2f}% / 理论")

    # 摩擦分解小标题
    st.caption("**📊 摩擦成本分解**（实际链上发生的钱）")
    c9, c10, c11 = st.columns(3)
    gas = float(metrics.total_gas_cost)
    slip = float(metrics.total_slippage_cost)
    lvr = float(metrics.total_lvr_cost)
    total = max(gas + slip + lvr, 0.001)
    c9.metric(
        "Gas", f"{gas:.2f}",
        delta=f"{gas / total * 100:.1f}% / 摩擦",
        help="按 (base_fee + priority_fee) × gas_limit_by_op 估算",
    )
    c10.metric(
        "Slippage", f"{slip:.2f}",
        delta=f"{slip / total * 100:.1f}% / 摩擦",
        help="阶梯函数：trade/tvl < 1% → 0.1%；< 5% → 0.3%；≥ 5% → 0.8%。"
             "小 TVL 池调仓滑点会显著高于大池。",
    )
    c11.metric(
        "LVR", f"{lvr:.2f}",
        delta=f"{lvr / total * 100:.1f}% / 摩擦",
        help="oracle vs pool 价差套利损耗；DefiLlama 不提供 pool 内部成交价，"
             "默认 oracle = pool_price 时 LVR = 0",
    )


def render_overview_tab(run: dict[str, Any]) -> None:
    render_metric_cards(run["metrics"], run["attribution"])
    st.plotly_chart(
        charts.nav_with_trade_markers(
            run["result"].nav_log, run["result"].trade_log,
            theoretical_nav=run["theoretical_nav"],
        ),
        use_container_width=True,
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("收益归因雷达")
        st.plotly_chart(
            charts.attribution_radar_multi({"当前": run["attribution"]}),
            use_container_width=True,
        )
        st.caption(
            f"理论最优收益 {float(run['attribution'].theoretical_total_return):,.2f} | "
            f"实际净收益 {float(run['attribution'].actual_return):,.2f}"
        )
    with right:
        st.subheader("HOLD 原因分布")
        trade_log = run["result"].trade_log
        holds = trade_log[trade_log["operation"] == "HOLD"] if not trade_log.empty else pd.DataFrame()
        if holds.empty:
            st.info("无 HOLD 记录")
        else:
            counts = holds["decision_reason"].value_counts().reset_index()
            counts.columns = ["原因", "次数"]
            st.dataframe(counts, use_container_width=True, hide_index=True)


def render_market_tab(run: dict[str, Any]) -> None:
    st.subheader("各池历史 APY")
    st.plotly_chart(charts.apy_history(run["pool_df"]), use_container_width=True)

    st.subheader("各池历史 TVL")
    st.plotly_chart(charts.tvl_history(run["pool_df"]), use_container_width=True)

    st.subheader("Gas 时序")
    st.plotly_chart(
        charts.gas_timeline(run["gas_df"], nav_log=run["result"].nav_log),
        use_container_width=True,
    )

    st.subheader("APY 热力图（池 × 时间）")
    st.plotly_chart(charts.apy_heatmap(run["pool_df"]), use_container_width=True)


def render_risk_tab(run: dict[str, Any]) -> None:
    # 自动诊断：根据持仓变化解释回撤形态
    nav_log = run["result"].nav_log
    pool_changes = (nav_log["pool_id"] != nav_log["pool_id"].shift()).sum()
    stable_days = 0
    if "pool_id" in nav_log.columns:
        for pid in nav_log["pool_id"].unique():
            if pid is None:
                continue
            # 简单启发：pool_id 含 USDC/USDT/DAI 视为稳定
            if any(s in str(pid).upper() for s in ["USDC", "USDT", "DAI", "USDS", "FRAX"]):
                stable_days += (nav_log["pool_id"] == pid).sum()

    st.subheader("回撤水下图")
    st.plotly_chart(
        charts.drawdown_underwater(nav_log),
        use_container_width=True,
    )
    if stable_days > 0:
        st.caption(
            f"💡 整段回测期中，策略约 **{stable_days} / {len(nav_log)} tick** 持有稳定币池"
            f"（token_price ≈ 1.0）；这段时间 NAV 仅由 APY 单调累计、无价格风险 → "
            f"回撤数学上为 0 是**预期行为**，反映策略成功避险。"
        )
    if pool_changes < 5:
        st.caption(
            f"⚠️ 整段只切换了 {pool_changes - 1} 次池子；试试**激进动量**预设观察高频调仓下的回撤。"
        )

    st.subheader("滚动夏普（30 日窗口）")
    st.plotly_chart(
        charts.rolling_sharpe(nav_log, window=30),
        use_container_width=True,
    )
    st.caption(
        "💡 稳定币持仓时 NAV 接近确定增长 → std → 0 → Sharpe 数学上 → ∞，"
        "图表已 clip 到 ±5 防止失真；标题会标注被截 tick 数。"
    )


def render_position_tab(run: dict[str, Any]) -> None:
    st.subheader("持仓时间线")
    st.plotly_chart(
        charts.position_timeline(run["result"].nav_log),
        use_container_width=True,
    )
    st.subheader("调仓明细")
    rotates = run["result"].trade_log[
        run["result"].trade_log["operation"] == "ROTATE"
    ].copy()
    if rotates.empty:
        st.info("整个回测期间未触发调仓。可尝试降低 threshold 或 τ-reset。")
    else:
        st.dataframe(
            rotates[[
                "tick", "timestamp", "from_pool_id", "to_pool_id",
                "amount", "gas_cost", "slippage_cost", "expected_gain",
            ]],
            use_container_width=True, hide_index=True,
        )


def render_cost_tab(run: dict[str, Any]) -> None:
    st.subheader("摩擦成本按时间堆积")
    st.plotly_chart(
        charts.cost_composition_stacked(run["result"].trade_log),
        use_container_width=True,
    )

    # 按目标池分组的摩擦汇总，让用户看出"哪个池最烧钱"
    trade_log = run["result"].trade_log
    rotates = trade_log[trade_log["operation"] == "ROTATE"] if not trade_log.empty else pd.DataFrame()
    if not rotates.empty:
        st.subheader("按调仓方向分组的摩擦汇总")
        grouped = rotates.groupby(["from_pool_id", "to_pool_id"], dropna=False).agg(
            次数=("amount", "count"),
            平均金额=("amount", "mean"),
            Gas总和=("gas_cost", "sum"),
            Slippage总和=("slippage_cost", "sum"),
            LVR总和=("lvr_cost", "sum"),
        ).reset_index()
        grouped["摩擦总和"] = grouped["Gas总和"] + grouped["Slippage总和"] + grouped["LVR总和"]
        grouped["平均滑点率"] = (grouped["Slippage总和"] / grouped["次数"] / grouped["平均金额"] * 100).round(3)
        st.dataframe(
            grouped[[
                "from_pool_id", "to_pool_id", "次数", "平均金额",
                "Gas总和", "Slippage总和", "LVR总和", "摩擦总和", "平均滑点率",
            ]].round(2),
            use_container_width=True, hide_index=True,
            column_config={
                "平均滑点率": st.column_config.NumberColumn(
                    "平均滑点率(%)", format="%.3f%%",
                    help="单次进/出该方向的滑点占交易额的百分比。"
                         "0.1% 表示该池 TVL 充足；0.3% 表示交易占池子 1%~5%；"
                         "0.8% 表示交易超过池子 5%（流动性紧张）",
                ),
            },
        )
        st.caption(
            "💡 滑点档位由 `trade_size / pool_TVL` 决定。如果某行平均滑点率 ≥ 0.3%，"
            "说明该池 TVL 偏小，频繁调仓会被滑点吃掉收益。"
        )

    st.subheader("Reinvest 日志")
    rein = run["result"].reinvest_log
    if rein.empty:
        st.info("无复投记录")
    else:
        st.dataframe(rein.tail(50), use_container_width=True, hide_index=True)
        st.caption(f"展示最近 50 条（共 {len(rein)} 条）")


def render_compare_tab(strategies: dict[str, dict]) -> None:
    if len(strategies) < 2:
        st.info("先在「概览」标签页运行 2 次以上回测并保存为命名策略，再来这里对比。")
        return

    st.subheader("NAV 曲线叠加")
    import plotly.graph_objects as go
    fig = go.Figure()
    for i, (name, s) in enumerate(strategies.items()):
        fig.add_trace(go.Scatter(
            x=s["result"].nav_log["timestamp"],
            y=s["result"].nav_log["nav"],
            mode="lines", name=name,
            line=dict(color=charts._color_for(i), width=2),
        ))
    fig.update_layout(
        height=420,
        xaxis_title="时间", yaxis_title="NAV",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("归因雷达对比")
    st.plotly_chart(
        charts.attribution_radar_multi({n: s["attribution"] for n, s in strategies.items()}),
        use_container_width=True,
    )

    st.subheader("指标对比表")
    rows = []
    for name, s in strategies.items():
        m, a = s["metrics"], s["attribution"]
        rows.append({
            "策略": name,
            "年化收益(%)": float(m.annualized_return) * 100,
            "夏普": float(m.sharpe_ratio),
            "索提诺": float(m.sortino_ratio),
            "MDD(%)": float(m.max_drawdown) * 100,
            "调仓数": a.rotation_count,
            "Gas 成本": float(m.total_gas_cost),
            "总摩擦": float(m.total_friction_cost),
            "实际收益": float(a.actual_return),
            "理论收益": float(a.theoretical_total_return),
        })
    st.dataframe(
        pd.DataFrame(rows).round(2),
        use_container_width=True, hide_index=True,
    )


def render_export(run: dict[str, Any]) -> None:
    st.subheader("导出 CSV")
    a, b, c = st.columns(3)
    attr_df = pd.DataFrame([run["attribution"].to_dict()])
    a.download_button(
        "归因报表",
        data=attr_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="attribution_report.csv", mime="text/csv",
        use_container_width=True,
    )
    b.download_button(
        "交易明细",
        data=run["result"].trade_log.to_csv(index=False).encode("utf-8-sig"),
        file_name="trade_log.csv", mime="text/csv",
        use_container_width=True,
    )
    c.download_button(
        "NAV 时序",
        data=run["result"].nav_log.to_csv(index=False).encode("utf-8-sig"),
        file_name="nav_log.csv", mime="text/csv",
        use_container_width=True,
    )


# =================================================================
# 主流程
# =================================================================

st.set_page_config(
    page_title="DeFi 收益轮动回测",
    page_icon="📈",
    layout="wide",
)

# session_state 初始化
if "strategies" not in st.session_state:
    st.session_state.strategies = {}
if "last_run" not in st.session_state:
    st.session_state.last_run = None

st.title("📈 DeFi 收益轮动 & 自动复投回测")
st.caption("策略参数全部可调；多 Tab 视图模仿专业量化平台；支持真实链上数据 + 多策略对比。")

# ----- Sidebar -----

with st.sidebar:
    st.header("数据源")
    data_source = st.radio(
        "选择数据来源", ["合成数据", "真实链上数据"],
        horizontal=True, label_visibility="collapsed",
    )

    if data_source == "真实链上数据":
        pool_csv_path = st.text_input(
            "Pool CSV 路径", value="data/real_pools.csv",
            help="先运行 `python fetch_data.py --pool-ids ... --pool-csv data/real_pools.csv`",
        )
        gas_csv_path = st.text_input(
            "Gas CSV 路径", value="data/real_gas.csv",
        )
        if not Path(pool_csv_path).exists():
            st.error(f"找不到 {pool_csv_path}")
        if not Path(gas_csv_path).exists():
            st.error(f"找不到 {gas_csv_path}")

    # ----- 回测时间区间 -----
    st.subheader("📅 回测时间区间")
    # 探测当前数据源的可用日期范围（缓存，避免重复加载）
    try:
        if data_source == "合成数据":
            _pool_probe, _ = _load_synthetic()
        else:
            if Path(pool_csv_path).exists() and Path(gas_csv_path).exists():
                _pool_probe, _ = _load_real_csvs_auto(pool_csv_path, gas_csv_path)
            else:
                _pool_probe = None

        if _pool_probe is not None and not _pool_probe.empty:
            _ts = pd.to_datetime(_pool_probe["timestamp"])
            min_d = _ts.min().date()
            max_d = _ts.max().date()
            st.caption(f"数据范围：{min_d} ~ {max_d}（共 {(max_d - min_d).days + 1} 天）")
            date_range = st.slider(
                "拖动选择回测起止日期",
                min_value=min_d, max_value=max_d,
                value=(min_d, max_d),
                format="YYYY-MM-DD",
                key="date_range",
                help="可裁掉数据头尾，只回测某段时间",
            )
            date_from, date_to = date_range
            days = (date_to - date_from).days + 1
            st.caption(f"⏱ 已选 **{days} 天**（{date_from} → {date_to}）")
        else:
            date_from = date_to = None
            st.warning("数据不存在，无法选择日期范围")
    except Exception as _e:
        date_from = date_to = None
        st.warning(f"无法读取日期范围：{_e}")

    st.divider()
    st.header("策略预设")

    # 预设选择：选中后把所有参数 widget 的 session_state 值刷新为 preset 默认
    preset_choice = st.selectbox(
        "快速载入",
        ["自定义"] + list_preset_names(),
        index=1,  # 默认载入"均衡（默认）"
        key="preset_choice",
        help="选预设会自动覆盖下方所有参数；调整任何滑块后回到自定义。",
    )

    if preset_choice != st.session_state.get("_active_preset"):
        # 用户切换了预设 → 把预设值写入各 widget 的 session_state key
        if preset_choice != "自定义":
            p = get_preset(preset_choice)
            st.session_state["sl_threshold"] = float(p.threshold)
            st.session_state["sl_tau"] = float(p.tau_reset)
            st.session_state["sl_cara"] = float(p.cara_alpha)
            st.session_state["sl_w_mom"] = float(p.weights.get("momentum", 0))
            st.session_state["sl_w_vol"] = float(p.weights.get("vol_penalty", 0))
            st.session_state["sl_w_mdd"] = float(p.weights.get("mdd_penalty", 0))
            st.session_state["sl_w_cara"] = float(p.weights.get("cara", 0))
            st.session_state["sl_w_pvol"] = float(p.weights.get("price_vol_penalty", 0))
            st.session_state["sl_w_pmdd"] = float(p.weights.get("price_mdd_penalty", 0))
            st.session_state["sl_slip"] = p.slip_choice
            st.session_state["sl_horizon"] = int(p.gain_horizon_ticks)
        st.session_state["_active_preset"] = preset_choice

    # 提示当前预设描述
    if preset_choice != "自定义":
        p = get_preset(preset_choice)
        with st.expander(f"📋 {p.name}: {p.description[:30]}...", expanded=False):
            st.write(p.description)
            st.caption(f"💡 {p.hint}")

    st.divider()
    st.header("策略参数")

    initial_capital = st.number_input(
        "初始资金", min_value=1_000.0, max_value=10_000_000.0,
        value=100_000.0, step=10_000.0,
    )
    top_n = st.slider("Top-N 候选池", 1, 10, 3)
    threshold = st.slider(
        "调仓增益阈值", 0.0001, 0.05, step=0.0001, format="%.4f",
        key="sl_threshold",
    )
    tau_reset = st.slider(
        "τ-reset 偏离度", 0.01, 0.20, step=0.01,
        key="sl_tau",
    )
    cara_alpha = st.slider(
        "CARA 风险厌恶 α", 0.5, 10.0, step=0.5,
        key="sl_cara",
    )
    gain_horizon = st.slider(
        "前瞻窗口 (tick)", 7, 180, step=7,
        key="sl_horizon",
        help="增益估算时向前看几个 tick；保守策略用更长窗口让 gain 跨过阈值",
    )

    st.subheader("多因子权重")
    w_mom = st.slider("动量", 0.0, 1.0, step=0.05, key="sl_w_mom")
    w_vol = st.slider("APY 波动率惩罚", 0.0, 1.0, step=0.05, key="sl_w_vol")
    w_mdd = st.slider("APY MDD 惩罚", 0.0, 1.0, step=0.05, key="sl_w_mdd")
    w_cara = st.slider("CARA 效用", 0.0, 1.0, step=0.05, key="sl_w_cara")
    w_pvol = st.slider(
        "价格波动率惩罚", 0.0, 1.0, step=0.05, key="sl_w_pvol",
        help="基于 token_price_series 的下行波动；捕捉本金贬值风险",
    )
    w_pmdd = st.slider(
        "价格 MDD 惩罚", 0.0, 1.0, step=0.05, key="sl_w_pmdd",
        help="基于 token_price_series 的最大回撤；惩罚历史价格暴跌池",
    )

    slip_choice = st.selectbox(
        "滑点档位", list(SLIPPAGE_PRESETS.keys()), key="sl_slip",
    )

    # 真实数据时不显示事件选项（事件 tick 与真实时间不一定对应）
    if data_source == "合成数据":
        selected_events = st.multiselect(
            "压力事件注入", list(EVENT_PRESETS.keys()), default=[],
        )
    else:
        selected_events = []
        st.caption("真实数据模式下默认不注入压力事件")

    st.divider()
    run_clicked = st.button("🚀 运行回测", type="primary", use_container_width=True)

    # 策略管理
    st.divider()
    st.header("策略管理")
    if st.session_state.last_run is not None:
        new_name = st.text_input("保存为命名策略", value="", placeholder="例如 A")
        if st.button("💾 保存当前结果", use_container_width=True):
            name = new_name.strip() or f"策略{len(st.session_state.strategies) + 1}"
            st.session_state.strategies[name] = copy.copy(st.session_state.last_run)
            st.success(f"已保存「{name}」")
    if st.session_state.strategies:
        st.write(f"**已保存策略**：{', '.join(st.session_state.strategies.keys())}")
        if st.button("🗑️ 清空所有策略", use_container_width=True):
            st.session_state.strategies.clear()
            st.rerun()


# ----- 触发回测 -----

if run_clicked:
    total_w = w_mom + w_vol + w_mdd + w_cara
    if total_w <= 0:
        st.error("权重总和必须 > 0，请至少给一个因子分配权重。")
        st.stop()
    if data_source == "真实链上数据":
        if not Path(pool_csv_path).exists() or not Path(gas_csv_path).exists():
            st.error("真实数据 CSV 不存在；请先运行 fetch_data.py 下载。")
            st.stop()

    params = {
        "data_source": "synthetic" if data_source == "合成数据" else "real",
        "pool_csv": pool_csv_path if data_source == "真实链上数据" else None,
        "gas_csv": gas_csv_path if data_source == "真实链上数据" else None,
        "date_from": date_from,
        "date_to": date_to,
        "initial_capital": initial_capital,
        "top_n": top_n,
        "threshold": threshold,
        "tau_reset": tau_reset,
        "cara_alpha": cara_alpha,
        "gain_horizon": gain_horizon,
        "w_mom": w_mom, "w_vol": w_vol, "w_mdd": w_mdd, "w_cara": w_cara,
        "w_pvol": w_pvol, "w_pmdd": w_pmdd,
        "slip_choice": slip_choice,
        "selected_events": selected_events,
        "preset_name": preset_choice,
    }
    with st.spinner("数据加载 + 引擎构建 + 主循环..."):
        st.session_state.last_run = run_backtest(params)
    st.success(f"回测完成（{st.session_state.last_run['result'].snapshots_processed} ticks）")


# ----- 主区域：多 Tab -----

if st.session_state.last_run is None and not st.session_state.strategies:
    st.info("👈 左侧配置参数后点击「运行回测」。")
    st.stop()

tabs = st.tabs(["📊 概览", "💹 行情", "⚠️ 风险", "🎯 仓位", "💸 成本", "🆚 多策略对比"])

run = st.session_state.last_run

with tabs[0]:
    if run is None:
        st.info("尚未运行回测；多策略可在最右标签页查看。")
    else:
        render_overview_tab(run)
        st.divider()
        render_export(run)

with tabs[1]:
    if run is None:
        st.info("先运行回测")
    else:
        render_market_tab(run)

with tabs[2]:
    if run is None:
        st.info("先运行回测")
    else:
        render_risk_tab(run)

with tabs[3]:
    if run is None:
        st.info("先运行回测")
    else:
        render_position_tab(run)

with tabs[4]:
    if run is None:
        st.info("先运行回测")
    else:
        render_cost_tab(run)

with tabs[5]:
    render_compare_tab(st.session_state.strategies)
