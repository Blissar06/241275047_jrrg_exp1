# 关键图表截图索引

本目录由 `python capture_screenshots.py` 一键生成。所有图表用 Plotly + Kaleido
导出为高分辨率 PNG（2× scale，1280px 宽起步），可直接拖入 PPT / 论文。

---

## 数据源 / 策略组合

| 第 N 组 | 数据 | 策略 | 用于演示 |
|---|---|---|---|
| 1 | 合成（365 天 × 3 池 / token_vol = [2%, 8%, 18%]） | BALANCED | 概览 + 行情 |
| 2 | **真实链上**（Maple_USDC / Vesper_ETH / Lido_stETH × 300 天） | CONSERVATIVE | 风险 + 仓位 + 成本 |
| 3 | 合成 × 5 个预设 | 5 策略全跑 | 多策略对比 |

---

## 图表清单（按 PPT 用途归类）

### 概览（步骤 1 演示）
| 文件 | 内容 | 看点 |
|---|---|---|
| `01_overview_nav.png` | NAV 曲线 + 调仓 ROTATE 菱形标记 + 理论最优虚线 | 策略 14 次调仓清晰可见；3 月底大涨后跟随，9 月跌破被动止损 |
| `02_overview_radar_single.png` | 单策略归因雷达 | 显示当前策略的 Gas/Slippage/LVR/调仓空窗占比 |

### 行情（步骤 2 演示）
| 文件 | 内容 | 看点 |
|---|---|---|
| `03_market_apy_history.png` | 3 池 APY 多线叠加 | 200~204 天 pool_B 闪崩可见 |
| `04_market_tvl_history.png` | 3 池 TVL 历史 | 数据来源真实性的可视化佐证 |
| `05_market_gas_timeline.png` | base_fee + priority_fee 时序堆积 | 150~154 天 spike × 5 一目了然 |
| `06_market_apy_heatmap.png` | 池 × 时间 APY 热力图 | pool_B 在 200 天附近有明显暗色格 |

### 风险（步骤 3 演示，**真实数据**）
| 文件 | 内容 | 看点 |
|---|---|---|
| `07_risk_drawdown.png` | 回撤水下图 | 仅入场摩擦造成 -2% 初始尖峰；之后 9 个月持仓 USDC，回撤=0（**模型避险成功的证据**） |
| `08_risk_rolling_sharpe.png` | 30 日滚动 Sharpe（clip ±5） | 稳定期间 Sharpe 极高（正常），波动期回落 |

### 仓位（步骤 3 演示，**真实数据**）
| 文件 | 内容 | 看点 |
|---|---|---|
| `09_position_timeline.png` | Gantt 持仓时间线 | 蓝色 Maple_USDC 占 99% 时间，橙色 Vesper_ETH 仅初始几天 |

### 成本（步骤 3 演示，**真实数据**）
| 文件 | 内容 | 看点 |
|---|---|---|
| `10_cost_stacked.png` | Gas/Slippage/LVR 堆积柱（按调仓时点） | 滑点远高于 Gas，反映 Vesper 池小的影响 |

### 多策略对比（步骤 4 演示）
| 文件 | 内容 | 看点 |
|---|---|---|
| `11_compare_nav_overlay.png` | 5 个预设 NAV 叠加 | **赢家**保守稳健/极端风险厌恶在 105k+，**输家**激进动量/低频价值跌到 87k 左右 |
| `12_compare_radar_multi.png` | 上：摩擦三分量雷达；下：调仓空窗 bar | 摩擦量级 0~35%，但**调仓空窗 0% vs 170%** 才是策略真正分水岭 |

---

## 动画演示

`demo.gif` — 10 秒循环动画，把 10 张代表性图按「概览 → 行情 → 风险 → 仓位 → 成本 → 多策略对比」顺序播一遍。

- 时长 10 秒（10 张图 × 1 秒/张）
- 503 KB（GIF 256 调色板压缩）
- 顶部 Tab 指示条高亮当前所在 Tab
- 底部副标题 + 进度条
- 适合插入 PPT 封面 / 答辩 demo 暖场

复现：
```bash
python capture_screenshots.py   # 先生成 12 张静态
python build_demo_gif.py        # 拼成 demo.gif
```

---

## 截图复现命令

```bash
# 重新生成全部 12 张
python capture_screenshots.py

# 重新拉真实链上数据（可选；CSV 已落盘）
python fetch_data.py --demo --days 300 \
                     --pool-csv data/real_pools.csv \
                     --gas-csv  data/real_gas.csv \
                     --gas-spike-start 150 --gas-spike-factor 5.0

# 跑全测试（覆盖 87.5%，30s 完成）
pytest tests/ --ignore=tests/perf
```

---

## 关键数据点（演示用，可直接说）

### 真实数据 / 保守稳健策略（图 7-10 的来源）
- 年化收益 **+8.74%**
- 最大回撤 **2.15%**
- Sharpe **1.540**
- 调仓次数 2
- 复投次数 ~290

### 合成数据 / 5 个预设对比（图 11-12 的来源）
| 策略 | 年化 | MDD | Sharpe | 调仓 | 调仓空窗 |
|---|---|---|---|---|---|
| **保守稳健** ✅ | **+8.82%** | **5.80%** | **1.076** | 19 | 0% |
| **极端风险厌恶** ✅ | +8.36% | 5.80% | 1.020 | 19 | 0% |
| 均衡（默认） | -3.65% | 20.69% | -0.230 | 14 | 107% |
| 激进动量 ❌ | -6.50% | 20.44% | -0.367 | 1 | 165% |
| 低频价值 ❌ | -11.71% | 21.57% | -0.781 | 16 | 170% |

**结论**：价格风险敏感的策略（保守 / 极端风险厌恶）在合成 + 真实数据上**双双跑赢**，
验证了 `TokenPriceVolPenaltyScorer` + `MarkToMarket` 的设计有效性。
