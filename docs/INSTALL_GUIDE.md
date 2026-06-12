# 助教评分操作指南 · 30 分钟跑通 Demo

> 对应评分项 **10 分 · README 清晰、报告规范、他人可复现**
> 目标：从 git clone 到看到完整看板 ≤ 30 分钟

---

## 🎯 你需要看到的最终效果

1. **325 个测试全部通过**（30 秒）
2. **端到端示例 7 项验收全过**（10 秒）
3. **Streamlit 看板能跑出真实数据回测**（启动 5 秒 + 回测 < 1 秒）

---

## 📦 环境要求

| 项 | 要求 | 验证 |
|---|---|---|
| 操作系统 | Windows / macOS / Linux 均可 | — |
| Python | **3.11 / 3.12 / 3.13** 任一即可 | `python --version` |
| 网络 | 可访问 GitHub、PyPI、`yields.llama.fi` | — |
| 磁盘 | < 100 MB（含依赖） | — |

> ⚠ Python 3.10 及以下**不行**，会因 `match-case`/`PEP 604` 等语法报错。

---

## 🚀 操作步骤（精确到秒）

### Step 1 · 克隆仓库（30 秒）

```bash
git clone https://github.com/Blissar06/241275047_jrrg_exp1.git
cd 241275047_jrrg_exp1
```

### Step 2 · 创建虚拟环境（30 秒，可选但推荐）

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows CMD
python -m venv venv
venv\Scripts\activate.bat

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 · 安装依赖（3 ~ 5 分钟）

```bash
pip install -r requirements.txt
```

**依赖清单**：numpy / pandas / pyarrow / pyyaml / streamlit / plotly / pytest / **hypothesis** / **pytest-cov** / **pytest-benchmark** —— 共 10 个包。

> ✅ 如果想重新生成 PPT/截图/GIF，再装 `pip install -r requirements-dev.txt`（含 python-pptx / lxml / kaleido / Pillow）；**但评分时不需要**，PPT 已经在 `docs/final_presentation.pptx`。

### Step 4 · 跑全套测试（30 秒）⭐

```bash
pytest tests/ --ignore=tests/perf
```

**期望输出**：

```
=========================== 325 passed in 30.46s ===========================
```

如果想看覆盖率：

```bash
pytest tests/ --ignore=tests/perf --cov --cov-report=term
```

期望 `TOTAL ... 87.5%`。

### Step 5 · 跑端到端示例（10 秒）⭐

```bash
python run_example.py
```

**期望输出**（最后一部分）：

```
=== 验收检查 ===
  [PASS] snapshots_processed == 365
  [PASS] runtime < 5s — 0.12s
  [PASS] nav_log len == 365
  [PASS] CSV Gas_Spike 期间 env_gas_base_fee 翻 5×
  [PASS] Pool_Exploit 后 Curve_3Pool 评分下跌
  [PASS] total_gas_cost > 0
  [PASS] total_lvr_cost > 0
=== 全部验收通过 ===
```

### Step 6 · 启动看板（5 秒）⭐

```bash
streamlit run ui/app.py
```

浏览器会自动打开 [http://localhost:8501](http://localhost:8501)。

**首次操作建议**：

1. 侧栏 **「数据源」** → 选 `真实链上数据`
   （`data/real_pools.csv` 已经预放好，3 池 300 天）
2. **「策略预设」** → 选 `保守稳健`
3. 点 **🚀 运行回测**
4. **概览 Tab** 应该看到：
   - 年化收益 **+8.74%**
   - 最大回撤 **2.15%**
   - Sharpe **1.540**
   - NAV 曲线 + 调仓菱形标记

---

## 🔬 评分项对应位置

| 评分项 | 怎么验证 | 关键文件 |
|---|---|---|
| **30 分** 金融逻辑正确 / 公式无误 | Step 5 全 PASS · 守恒等式误差 < 0.001% | `report/attribution.py` + `tests/test_attribution.py` |
| **25 分** 仿真闭环 / 交互可视化 | Step 6 看板 6 Tab 全部可用 | `ui/app.py` |
| **20 分** 模块化 / 测试 / 性能 | Step 4 全 325 通过 + 覆盖率 87.5% | `tests/TEST_PLAN.md` |
| **15 分** 场景设计 / 对比分析 | 看板「多策略对比 Tab」+ `blackbox/test_scenarios.py` | `tests/blackbox/test_scenarios.py` |
| **10 分** README / 可复现 | 你**正在看**这份指南 + README.md | 本文件 |

---

## 🐛 常见问题与解决

### Q1 · `pip install` 装到一半报错

**可能原因**：网络慢 / PyPI 镜像不稳

**解决**：换清华镜像
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2 · `pytest` 报 `ModuleNotFoundError: No module named 'hypothesis'`

**说明**：依赖没装齐。重新跑：
```bash
pip install -r requirements.txt
```
（要装最新的 requirements.txt，里面已经包含 hypothesis）

### Q3 · `streamlit run` 浏览器没自动打开

手动打开 [http://localhost:8501](http://localhost:8501) 即可。

### Q4 · 看板提示「找不到 data/real_pools.csv」

如果不小心删了真实数据 CSV，重新拉：
```bash
python fetch_data.py --demo --days 300
```
30 秒完成。需要网络能访问 `yields.llama.fi`。

### Q5 · 测试一直卡在某个 `test_perf_smoke...`

`tests/perf/` 是性能基准，慢一些（约 10 秒）。**评分不需要跑这个**，加 `--ignore=tests/perf` 即可：
```bash
pytest tests/ --ignore=tests/perf
```

### Q6 · Streamlit 看板的真实数据日期范围 slider 没出现

刷新一次（**Ctrl + F5** 或 **R** 键）即可——Streamlit 改动需要硬刷新。

---

## ⏱ 时间预算

| 步骤 | 预计耗时 |
|---|---|
| Git clone | 30 s |
| 创建 venv | 30 s |
| pip install | **3 ~ 5 min**（最慢的一步） |
| pytest 全测试 | 30 s |
| run_example | 10 s |
| streamlit 启动 + 操作 | 1 min |
| **合计** | **6 ~ 8 分钟** |

远低于 30 分钟预算。

---

## 📚 参考材料

- [README.md](../README.md) · 项目总览
- [tests/TEST_PLAN.md](../tests/TEST_PLAN.md) · 完整测试矩阵
- [docs/screenshots/](screenshots/) · 12 张关键图 + demo.gif
- [docs/final_presentation.pptx](final_presentation.pptx) · 答辩 PPT
- [docs/presentation_script.md](presentation_script.md) · 答辩讲稿

如评分时遇到任何问题，请联系组长 **Xingrui Zhao**。

---

## ✅ 验收 Checklist

> 助教可逐项打勾

- [ ] Step 1 ~ 3 完成，环境就绪
- [ ] Step 4 · `pytest tests/ --ignore=tests/perf` 输出 `325 passed`
- [ ] Step 5 · `python run_example.py` 末尾输出 `=== 全部验收通过 ===`
- [ ] Step 6 · Streamlit 看板成功打开，能跑出保守稳健策略 8.74% 年化的结果
- [ ] 看了 docs/ 下的 PPT 与截图
- [ ] 看了 tests/TEST_PLAN.md 的测试矩阵

如果以上全 ✅，本项目满足课程评分标准 10 分的「他人可复现」要求。
