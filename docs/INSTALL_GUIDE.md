# 操作指南 

## 📦 环境要求

| 项 | 要求 | 验证 |
|---|---|---|
| 操作系统 | Windows / macOS / Linux 均可 | — |
| Python | **3.11 / 3.12 / 3.13** 任一即可 | `python --version` |
| 网络 | 可访问 GitHub、PyPI、`yields.llama.fi` | — |
| 磁盘 | < 100 MB（含依赖） | — |

> ⚠ Python 3.10 及以下**不行**，会因 `match-case`/`PEP 604` 等语法报错。

---

## 🚀 操作步骤

### Step 1 · 克隆仓库

```bash
git clone https://github.com/Blissar06/241275047_jrrg_exp1.git
cd 241275047_jrrg_exp1
```

### Step 2 · 创建虚拟环境

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

### Step 3 · 安装依赖

```bash
pip install -r requirements.txt
```

**依赖清单**：numpy / pandas / pyarrow / pyyaml / streamlit / plotly / pytest / **hypothesis** / **pytest-cov** / **pytest-benchmark** —— 共 10 个包。



### Step 4 · 跑全套测试⭐

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

### Step 5 · 跑端到端示例⭐

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

### Step 6 · 启动看板⭐

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





---

## 📚 参考材料

- [README.md](../README.md) · 项目总览
- [tests/TEST_PLAN.md](../tests/TEST_PLAN.md) · 完整测试矩阵
- [docs/screenshots/](screenshots/) · 12 张关键图 + demo.gif
- [docs/final_presentation.pptx](final_presentation.pptx) · 答辩 PPT
- [docs/presentation_script.md](presentation_script.md) · 答辩讲稿

如评分时遇到任何问题，请联系组长 **Xingrui Zhao**。

---

