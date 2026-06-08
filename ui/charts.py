"""图表工厂模块：纯函数 → Plotly Figure。

设计目标：
  - 每个函数无副作用、不依赖 streamlit
  - 输入是 pd.DataFrame / pd.Series，输出是 plotly.graph_objects.Figure
  - 易于单元测试与跨页面复用

图表清单：
  1. nav_with_trade_markers       NAV + 调仓事件标记
  2. apy_history                   各池历史 APY 多线叠加
  3. tvl_history                   各池历史 TVL 多线叠加
  4. gas_timeline                  Gas base+priority fee 时序
  5. drawdown_underwater           回撤水下图
  6. position_timeline             持仓时间线（Gantt 风格）
  7. apy_heatmap                   APY 热力图（池 × 时间）
  8. rolling_sharpe                滚动夏普
  9. cost_composition_stacked      Gas/Slippage/LVR 堆积柱
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# 调色板：对色盲友好（D3 默认配色基础上微调）
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]


def _color_for(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


def _empty_fig(message: str = "无数据") -> go.Figure:
    """构造一个带提示文字的空图，避免下游 None 处理。"""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# =================================================================
# 1. NAV + 调仓标记
# =================================================================

def nav_with_trade_markers(
    nav_log: pd.DataFrame,
    trade_log: pd.DataFrame,
    theoretical_nav: Optional[pd.Series] = None,
) -> go.Figure:
    """主图：NAV 曲线 + 调仓事件菱形标记 + 可选理论最优叠加。"""
    if nav_log is None or nav_log.empty:
        return _empty_fig("无 NAV 数据")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav_log["timestamp"], y=nav_log["nav"],
        mode="lines", name="实际 NAV",
        line=dict(color=_PALETTE[0], width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>NAV=%{y:,.2f}<extra></extra>",
    ))

    if theoretical_nav is not None and not theoretical_nav.empty:
        n = min(len(theoretical_nav), len(nav_log))
        fig.add_trace(go.Scatter(
            x=nav_log["timestamp"].iloc[:n],
            y=theoretical_nav.iloc[:n].values,
            mode="lines", name="理论最优（无摩擦）",
            line=dict(color=_PALETTE[1], width=1.5, dash="dash"),
        ))

    # 调仓菱形标记
    if trade_log is not None and not trade_log.empty and "operation" in trade_log.columns:
        rotates = trade_log[trade_log["operation"] == "ROTATE"]
        if not rotates.empty:
            # 把 trade_log 与 nav_log 用 tick 对齐取 NAV 值
            nav_indexed = nav_log.set_index("tick")["nav"]
            rotates = rotates[rotates["tick"].isin(nav_indexed.index)]
            marker_y = nav_indexed.loc[rotates["tick"]].values
            hover = [
                f"tick={int(t)}<br>{f or '现金'} → {to}<br>amount={a:,.0f}"
                for t, f, to, a in zip(
                    rotates["tick"], rotates["from_pool_id"],
                    rotates["to_pool_id"], rotates["amount"],
                )
            ]
            fig.add_trace(go.Scatter(
                x=rotates["timestamp"], y=marker_y,
                mode="markers", name="ROTATE",
                marker=dict(symbol="diamond", size=10, color=_PALETTE[3], line=dict(width=1, color="white")),
                text=hover, hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_layout(
        height=420,
        xaxis_title="时间", yaxis_title="NAV",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# =================================================================
# 2. APY 历史
# =================================================================

def apy_history(pool_df: pd.DataFrame) -> go.Figure:
    """各池历史 APY 多线叠加。pool_df 需含 timestamp / pool_id / apy。"""
    if pool_df is None or pool_df.empty:
        return _empty_fig("无 APY 数据")

    fig = go.Figure()
    for i, pid in enumerate(sorted(pool_df["pool_id"].unique())):
        sub = pool_df[pool_df["pool_id"] == pid].sort_values("timestamp")
        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["apy"] * 100,  # 百分比显示
            mode="lines", name=pid,
            line=dict(color=_color_for(i), width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>" + pid + " APY=%{y:.2f}%<extra></extra>",
        ))

    fig.update_layout(
        height=380,
        xaxis_title="时间", yaxis_title="APY (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# =================================================================
# 3. TVL 历史
# =================================================================

def tvl_history(pool_df: pd.DataFrame) -> go.Figure:
    if pool_df is None or pool_df.empty:
        return _empty_fig("无 TVL 数据")

    fig = go.Figure()
    for i, pid in enumerate(sorted(pool_df["pool_id"].unique())):
        sub = pool_df[pool_df["pool_id"] == pid].sort_values("timestamp")
        fig.add_trace(go.Scatter(
            x=sub["timestamp"], y=sub["tvl"],
            mode="lines", name=pid,
            line=dict(color=_color_for(i), width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>" + pid + " TVL=%{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(
        height=380,
        xaxis_title="时间", yaxis_title="TVL",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# =================================================================
# 4. Gas 时序
# =================================================================

def gas_timeline(
    gas_df: pd.DataFrame,
    nav_log: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Gas base_fee + priority_fee 时序。优先用 nav_log 里的 env 字段，
    否则用 gas_df。"""
    if nav_log is not None and "env_gas_base_fee" in nav_log.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nav_log["timestamp"], y=nav_log["env_gas_base_fee"],
            mode="lines", name="base_fee", line=dict(color=_PALETTE[0]),
            stackgroup="g",
        ))
        fig.add_trace(go.Scatter(
            x=nav_log["timestamp"], y=nav_log["env_gas_priority_fee"],
            mode="lines", name="priority_fee", line=dict(color=_PALETTE[1]),
            stackgroup="g",
        ))
    elif gas_df is not None and not gas_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gas_df["timestamp"], y=gas_df["base_fee"],
            mode="lines", name="base_fee", line=dict(color=_PALETTE[0]),
            stackgroup="g",
        ))
        fig.add_trace(go.Scatter(
            x=gas_df["timestamp"], y=gas_df["priority_fee"],
            mode="lines", name="priority_fee", line=dict(color=_PALETTE[1]),
            stackgroup="g",
        ))
    else:
        return _empty_fig("无 Gas 数据")

    fig.update_layout(
        height=320,
        xaxis_title="时间", yaxis_title="Gas (计价本位 / gas-unit)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# =================================================================
# 5. Drawdown underwater
# =================================================================

def drawdown_underwater(nav_log: pd.DataFrame) -> go.Figure:
    """水下图：当前回撤 = nav / cummax - 1（负值），显示为面积。"""
    if nav_log is None or nav_log.empty or "nav" not in nav_log.columns:
        return _empty_fig("无 NAV 数据")

    nav = nav_log["nav"].values.astype(float)
    peaks = np.maximum.accumulate(nav)
    dd = np.where(peaks > 0, nav / peaks - 1.0, 0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav_log["timestamp"], y=dd * 100,
        mode="lines", name="回撤",
        line=dict(color=_PALETTE[3], width=1),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.25)",
        hovertemplate="%{x|%Y-%m-%d}<br>回撤=%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        xaxis_title="时间", yaxis_title="回撤 (%)",
        # 数据天然 ≤ 0，让 Plotly 自动缩放即可；只确保 0 是上边界附近
        yaxis=dict(zeroline=True, zerolinecolor="gray", zerolinewidth=1),
        showlegend=False,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


# =================================================================
# 6. 持仓时间线（Gantt 风格）
# =================================================================

def position_timeline(nav_log: pd.DataFrame) -> go.Figure:
    """根据 nav_log['pool_id'] 的连续段构造 Gantt（用 px.timeline 渲染）。"""
    if nav_log is None or nav_log.empty or "pool_id" not in nav_log.columns:
        return _empty_fig("无持仓数据")

    df = nav_log[["timestamp", "pool_id"]].copy()
    df["pool_id"] = df["pool_id"].fillna("现金").replace({None: "现金"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["change"] = (df["pool_id"] != df["pool_id"].shift()).cumsum()

    # 推断每 tick 的步长以便段末尾延伸一格（否则末日为闭区间会缺一天）
    if len(df) >= 2:
        step = df["timestamp"].diff().median()
    else:
        step = pd.Timedelta(days=1)

    segments = []
    for _, group in df.groupby("change"):
        start = group["timestamp"].iloc[0]
        end = group["timestamp"].iloc[-1] + step
        segments.append({
            "pool_id": group["pool_id"].iloc[0],
            "start": start,
            "end": end,
            "days": len(group),
        })

    if not segments:
        return _empty_fig("无持仓段")

    seg_df = pd.DataFrame(segments)
    unique_pools = sorted(seg_df["pool_id"].unique())
    color_map = {p: _color_for(i) for i, p in enumerate(unique_pools)}

    import plotly.express as px
    fig = px.timeline(
        seg_df,
        x_start="start", x_end="end", y="pool_id",
        color="pool_id",
        color_discrete_map=color_map,
        hover_data={"days": True, "start": True, "end": True, "pool_id": False},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=max(180, 60 * len(unique_pools) + 80),
        xaxis_title="时间", yaxis_title="持仓池",
        showlegend=False,
        margin=dict(l=80, r=20, t=30, b=40),
    )
    return fig


# =================================================================
# 7. APY 热力图
# =================================================================

def apy_heatmap(pool_df: pd.DataFrame) -> go.Figure:
    """池 × 时间的 APY 热力图。"""
    if pool_df is None or pool_df.empty:
        return _empty_fig("无 APY 数据")

    pivot = pool_df.pivot_table(
        index="pool_id", columns="timestamp", values="apy",
        aggfunc="mean",
    ).sort_index()
    if pivot.empty:
        return _empty_fig("无 APY 数据")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,       # 百分比
        x=pivot.columns,
        y=pivot.index,
        colorscale="Viridis",
        colorbar=dict(title="APY (%)"),
        hovertemplate="池=%{y}<br>时间=%{x|%Y-%m-%d}<br>APY=%{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(240, 40 * len(pivot.index) + 80),
        xaxis_title="时间", yaxis_title="池",
        margin=dict(l=80, r=20, t=30, b=40),
    )
    return fig


# =================================================================
# 8. 滚动夏普
# =================================================================

def rolling_sharpe(
    nav_log: pd.DataFrame,
    window: int = 30,
    periods_per_year: int = 365,
    display_clip: float = 5.0,
) -> go.Figure:
    """rolling Sharpe（窗口默认 30 天）。

    显示限幅：
      - std < 1e-6（持仓接近确定收益）→ Sharpe 置 0
      - 真实 Sharpe 经 clip 到 ±display_clip（默认 5）；
        因为对几乎无风险的稳定币策略数学上 Sharpe → ∞，但图表显示意义不大
    """
    if nav_log is None or nav_log.empty or "nav" not in nav_log.columns:
        return _empty_fig("无 NAV 数据")
    if len(nav_log) < window + 1:
        return _empty_fig(f"数据不足 {window + 1} 个 tick")

    nav = nav_log["nav"].astype(float)
    rets = nav.pct_change()
    mean = rets.rolling(window).mean()
    std = rets.rolling(window).std(ddof=0)
    # 1) 极小 std → 0（无意义的"无风险"窗口）
    sharpe_raw = np.where(std > 1e-6, mean / std * (periods_per_year ** 0.5), 0.0)
    # 2) clip 到 ±display_clip
    sharpe = np.clip(sharpe_raw, -display_clip, display_clip)
    clipped_count = int(np.sum(np.abs(sharpe_raw) > display_clip))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav_log["timestamp"], y=sharpe,
        mode="lines", name=f"Sharpe (window={window})",
        line=dict(color=_PALETTE[2], width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Sharpe=%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    title_suffix = ""
    if clipped_count > 0:
        title_suffix = f" — {clipped_count} 个 tick 被截到 ±{display_clip}（稳定收益期）"
    fig.update_layout(
        title=dict(text=f"窗口={window} 天{title_suffix}", x=0.02, font=dict(size=11)),
        height=280,
        xaxis_title="时间", yaxis_title="Sharpe (clipped)",
        yaxis=dict(range=[-display_clip * 1.1, display_clip * 1.1]),
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# =================================================================
# 9. 摩擦成本堆积
# =================================================================

def cost_composition_stacked(trade_log: pd.DataFrame) -> go.Figure:
    """按 tick 堆积 gas / slippage / lvr（仅 ROTATE 行）。"""
    if trade_log is None or trade_log.empty or "operation" not in trade_log.columns:
        return _empty_fig("无交易数据")
    rotates = trade_log[trade_log["operation"] == "ROTATE"].copy()
    if rotates.empty:
        return _empty_fig("无 ROTATE 记录")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rotates["timestamp"], y=rotates["gas_cost"],
        name="Gas", marker=dict(color=_PALETTE[0]),
    ))
    fig.add_trace(go.Bar(
        x=rotates["timestamp"], y=rotates["slippage_cost"],
        name="Slippage", marker=dict(color=_PALETTE[1]),
    ))
    fig.add_trace(go.Bar(
        x=rotates["timestamp"], y=rotates["lvr_cost"],
        name="LVR", marker=dict(color=_PALETTE[3]),
    ))

    fig.update_layout(
        height=320,
        barmode="stack",
        xaxis_title="时间", yaxis_title="成本（计价本位）",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# =================================================================
# 辅助：归因雷达（单/多策略可用）
# =================================================================

def attribution_radar_multi(strategies: dict) -> go.Figure:
    """strategies: {name: AttributionReport} → 摩擦三分量雷达 + 调仓空窗 bar 复合图。

    设计：Gas / Slippage / LVR 是「实际花掉的钱」（量级 0~10%），调仓空窗
    是「机会成本」（量级 0~200%），两者不在同一物理意义上，混在一张雷达里
    会导致前者被压缩到中心。这里拆成上下两个 subplot：
      - 上：3 维摩擦雷达，半径轴自适应 max
      - 下：调仓空窗水平条形图
    """
    if not strategies:
        return _empty_fig("无策略数据")

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        specs=[[{"type": "polar"}], [{"type": "xy"}]],
        subplot_titles=("摩擦成本三分量（占理论收益百分比）", "调仓空窗折损占比"),
        vertical_spacing=0.15,
    )

    # ---- 上：摩擦雷达（Gas / Slippage / LVR）----
    friction_categories = ["Gas", "Slippage", "LVR"]
    max_friction = 0.0
    for name, attr in strategies.items():
        max_friction = max(
            max_friction,
            float(attr.gas_cost_pct),
            float(attr.slippage_pct),
            float(attr.lvr_pct),
        )
    radial_range = max(max_friction * 1.2, 1.0)  # 至少 1%，避免太小

    for i, (name, attr) in enumerate(strategies.items()):
        vals = [
            float(attr.gas_cost_pct),
            float(attr.slippage_pct),
            float(attr.lvr_pct),
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=friction_categories + [friction_categories[0]],
                fill="toself",
                name=name,
                line=dict(color=_color_for(i)),
                opacity=0.55,
                hovertemplate="%{theta}=%{r:.3f}%<extra>" + name + "</extra>",
                legendgroup=name,
            ),
            row=1, col=1,
        )

    # ---- 下：调仓空窗 bar ----
    names = list(strategies.keys())
    idle_values = [float(strategies[n].rotation_idle_pct) for n in names]
    fig.add_trace(
        go.Bar(
            x=idle_values,
            y=names,
            orientation="h",
            marker=dict(color=[_color_for(i) for i in range(len(names))]),
            text=[f"{v:.1f}%" for v in idle_values],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{y}<br>调仓空窗=%{x:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=620,
        polar=dict(radialaxis=dict(
            visible=True, ticksuffix="%",
            range=[0, radial_range],
        )),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
        margin=dict(l=80, r=20, t=60, b=40),
    )
    fig.update_xaxes(title="调仓空窗占理论收益 %", row=2, col=1, ticksuffix="%")
    return fig
