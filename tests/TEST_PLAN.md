# 测试计划 — DeFi 收益轮动回测平台

根据《金融软件工程》第 8 讲《测试基础》和《白盒黑盒测试》两份课件设计的系统化测试方案。

---

## 测试金字塔

```
                    ┌─────────────────────┐
                    │   性能基准 (perf)    │  10 个 benchmark
                    │   pytest-benchmark   │  (跑法独立, --benchmark-only)
                    └─────────────────────┘
                  ┌─────────────────────────┐
                  │   场景/集成测试 (E2E)    │  21 个测试
                  │   黑盒场景 + 完整管道    │  (test_integration.py + scenarios)
                  └─────────────────────────┘
              ┌───────────────────────────────┐
              │  黑盒：等价类/边界/决策表       │  77 个测试
              │  blackbox/  (按方法学分目录)   │
              └───────────────────────────────┘
          ┌─────────────────────────────────────┐
          │  白盒：路径覆盖 / 条件组合           │  41 个测试
          │  whitebox/                          │
          └─────────────────────────────────────┘
      ┌───────────────────────────────────────────┐
      │  属性测试 (hypothesis)                    │  11 个属性
      │  property/  (每个 50+ 随机用例)            │
      └───────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────┐
  │  单元测试 — 按模块分                              │  185 个测试
  │  test_*.py  (已有基线)                           │
  └─────────────────────────────────────────────────┘
```

**汇总**：
- 单元 / 集成 / 场景 / 黑盒 / 白盒 / 属性：**325 个测试**
- 性能基准：**10 个 benchmark**
- 总覆盖率：**87.5%**（branch coverage 启用）

---

## 测试方法学到 PPT 章节的映射

### 黑盒测试（PPT 第 9 讲 §9.2）

| PPT 方法 | 实现位置 | 用例数 | 编号前缀 |
|---|---|---|---|
| **等价类划分** (§9.2 等价分类法) | `tests/blackbox/test_equivalence_classes.py` | 31 | `EC-<函数>-V/I<n>` |
| **边界值分析** (§9.2 边界值分析法) | `tests/blackbox/test_boundary_values.py` | 34 | `BV-<函数>-<nn>` |
| **决策表法** (§9.2 决策表法) | `tests/blackbox/test_decision_tables.py` | 12 | `DT-RE/RI-<nn>` |
| **场景法** (§9.2 场景测试) | `tests/blackbox/test_scenarios.py` | 11 | `SC-<场景>-<nn>` |

### 白盒测试（PPT 第 9 讲 §一）

| PPT 方法 | 实现位置 | 用例数 | 编号前缀 |
|---|---|---|---|
| **基本路径覆盖** (§一·6 基本路径覆盖) | `tests/whitebox/test_path_coverage.py` | 22 | `PATH-<函数>-<nn>` |
| **条件组合覆盖** (§一·5 条件组合覆盖) | `tests/whitebox/test_condition_combinations.py` | 19 | `CC-<判定>-<nn>` |
| **分支覆盖** (§一·4 判定覆盖) | 集成测试隐式覆盖 | — | coverage.py 度量 |

### 其他类型（PPT 第 8 讲）

| PPT 分类 | 实现 | 数量 |
|---|---|---|
| **单元测试** | `tests/test_*.py`（基线） | 185 |
| **集成测试** | `tests/test_integration.py` + `scenarios` | 19 |
| **回归测试** | 全套测试每次跑 | 325 |
| **性能测试** | `tests/perf/test_benchmarks.py` | 10 benchmarks |
| **冒烟测试** | 标记 `@pytest.mark.perf` 的 1k tick 用例 | 2 |

---

## 测试用例编号规范

| 前缀 | 含义 | 示例 |
|---|---|---|
| `EC-<FN>-V<n>` | 等价类：有效类（Valid）第 n 个 | `EC-SLIP-V2` |
| `EC-<FN>-I<n>` | 等价类：无效类（Invalid）第 n 个 | `EC-EWM-I1` |
| `BV-<FN>-<nn>` | 边界值 | `BV-SLIP-05` |
| `DT-RE-<nn>` | 决策表：RotationEngine 行号 | `DT-RE-04` |
| `DT-RI-<nn>` | 决策表：ReinvestEngine 行号 | `DT-RI-03` |
| `SC-<scn>-<nn>` | 场景：业务流程 | `SC-EXPL-02` |
| `PATH-<FN>-<nn>` | 路径覆盖 | `PATH-RE-05` |
| `CC-<判定>-<nn>` | 条件组合 | `CC-NEG-04` |
| `PROP-<不变量>-<nn>` | 属性测试 | `PROP-MDD-01` |
| `PERF-<对象>-<nn>` | 性能基准 | `PERF-RUN-01` |

**函数缩写表**（出现在 EC / BV / PATH 中）：
- `CAP` = apply_capacity_decay
- `SLIP` = estimate_slippage
- `EWM` = ewma
- `GATE` = RotationEngine._gate
- `RE` = RotationEngine.evaluate
- `RI` = ReinvestEngine 系列
- `FE` = FrictionEstimator.estimate
- `TR` = _check_tau_reset
- `MDD` = max_drawdown
- `SHARP` / `SORT` = sharpe_ratio / sortino_ratio
- `AR` = annualized_return
- `WCFG` = WeightConfig

---

## 怎么跑

```bash
# 全部测试（不含性能基准，约 30s）
pytest tests/ --ignore=tests/perf

# 按方法学分组跑
pytest -m blackbox            # 88 个黑盒
pytest -m whitebox            # 41 个白盒
pytest -m property            # 11 个属性测试（每个 50+ 随机用例）
pytest -m scenario            # 11 个场景测试
pytest -m equivalence         # 31 个等价类
pytest -m boundary            # 34 个边界值
pytest -m decision_table      # 12 个决策表
pytest -m path                # 22 个路径
pytest -m condition           # 19 个条件组合

# 覆盖率报告
pytest --cov --cov-report=html         # 生成 htmlcov/index.html
pytest --cov --cov-report=term-missing # 终端显示 + 缺失行号

# 性能基准（独立跑）
pytest tests/perf --benchmark-only

# 性能基准 + 保存基线
pytest tests/perf --benchmark-only --benchmark-save=baseline

# 性能基准回归检测（CI 集成）
pytest tests/perf --benchmark-only --benchmark-compare=baseline \
       --benchmark-compare-fail=mean:20%

# 单文件
pytest tests/blackbox/test_decision_tables.py -v
pytest tests/whitebox/test_path_coverage.py::TestRotationEvaluatePaths -v
```

---

## 覆盖率目标与实测

`.coveragerc` 启用 **branch coverage**（不仅看语句，也看分支）。

### 当前覆盖（2026-05-29）

| 模块 | 语句覆盖率 | 缺漏行（关键） |
|---|---|---|
| `report/attribution.py` | **100.0%** | — |
| `strategy/scorers/momentum.py` | **100.0%** | — |
| `strategy/scorers/risk_penalty.py` | **100.0%** | — |
| `strategy/scoring_engine.py` | 98.4% | — |
| `strategy/interfaces.py` | 96.5% | — |
| `report/metrics.py` | 93.1% | 个别边界 |
| `backtest/engine.py` | 92.5% | persist 错误路径 |
| `strategy/reinvest_engine.py` | 94.4% | — |
| `strategy/rotation_engine.py` | 91.6% | — |
| `data_model/asset.py` | 90.0% | — |
| `backtest/event_injector.py` | 90.1% | 未知事件类型 |
| `data/sample_data.py` | 88.9% | — |
| `backtest/cost_model.py` | 88.0% | LVR 异常路径 |
| `data_model/loader.py` | 86.2% | 罕见数据格式 |
| `strategy/scorers/cara.py` | 83.8% | — |
| `strategy/scorers/_common.py` | 84.0% | — |
| `strategy/presets.py` | 84.4% | error 分支 |
| `data_model/preprocessor.py` | 76.3% | 部分罕见分支 |
| `strategy/gain_estimator.py` | 73.0% | drift 边界 |
| `data/onchain_fetcher.py` | 68.8% | 真实 HTTP（生产用，单测靠 mock） |
| **TOTAL** | **87.5%** | branch=82.4% |

### 目标
- **总覆盖率 ≥ 85%**（已达成）
- **核心策略模块 ≥ 90%**（已达成）
- `gain_estimator` / `preprocessor` 提到 85%+（待后续补充罕见分支用例）

---

## 缺陷分类与跟踪（PPT 第 8 讲 §缺陷分类）

按 PPT 分类，本项目在测试过程中发现并修复的缺陷：

| 缺陷 ID | 分类 | 触发用例 | 描述 | 状态 |
|---|---|---|---|---|
| BUG-001 | 边界条件 | `test_two_runs_produce_identical_outputs` | DataFrame 含 None 列时 `==` 比较返回 False | 已修：用 `.equals()` |
| BUG-002 | 数据完整性 | `test_data_integrity_error_fallbacks_to_hold` | `_hold()` 覆盖 ERROR 状态 | 已修：调整 state 赋值顺序 |
| BUG-003 | 模型缺陷 | `test_sharpe_zero_when_no_volatility` | 浮点累积让恒定收益的 std=1e-17，Sharpe=1e29 | 已修：epsilon < 1e-12 视为 0 |
| BUG-004 | **属性测试发现** | `test_PROP_DET_01_same_synthetic_seed_same_nav` | `generate_sample_data(n_days<200)` crash_window 越界 | 已修：clip 切片 |
| BUG-005 | 索引对齐 | 手测真实数据 | `df.tail()` 保留原索引，Series 赋值按索引对齐 → NaN | 已修：`.reset_index(drop=True) + .values` |
| BUG-006 | 数据完整性 | 手测真实数据 | DefiLlama 跨池 timestamp 有秒级偏移 | 已修：`.dt.floor("D")` 归一化 |
| BUG-007 | 缓存失效 | 手测 UI | `@st.cache_data` 缓存 CSV 路径，文件内容变了不重读 | 已修：缓存键加 file mtime |

---

## CI 集成（PPT 第 8 讲 §自动化）

`.github/workflows/ci.yml`：

- 触发：`push to main/dev/frosty-*`、`PR to main`
- Python 矩阵：3.11 / 3.12 / 3.13
- 步骤：
  1. 安装依赖（缓存 pip）
  2. `compileall` 语法预检
  3. `pytest --cov` 全套测试 + 覆盖率
  4. 性能基准（不阻断）
  5. 覆盖率上传到 Codecov（仅 Python 3.13）
  6. `ruff check` 风格检查（informational）

---

## 还能扩展什么（未做但可加）

| 项 | 优先级 | 工作量 |
|---|---|---|
| 突变测试 (mutation testing, `mutmut`) | 中 | 1 天 |
| 模糊测试 (libFuzzer / atheris) 针对 loader | 低 | 0.5 天 |
| UI E2E 测试（Playwright + Streamlit） | 中 | 1-2 天 |
| 压力测试：10k tick × 50 池 内存/时延曲线 | 低 | 0.5 天 |
| Locust 负载测试 fetcher（DefiLlama API rate limit） | 低 | 0.5 天 |
| 安全测试：CSV 注入、路径穿越 | 中 | 0.5 天 |

---

## 编写新测试的 checklist

每加一个测试，按 PPT《测试基础》§测试用例三要素：

- [ ] **前置条件**：`fixture` 或函数开头明确写出测试输入的环境约束
- [ ] **输入**：参数化（pytest.mark.parametrize 或 hypothesis）尽量覆盖等价类
- [ ] **预期输出**：assert 写明业务含义，不要只比相等
- [ ] **方法学标注**：用 `@pytest.mark.<blackbox|whitebox|property|...>` 分类
- [ ] **编号**：函数名按 `test_<PREFIX>_<NN>_<描述>` 约定
- [ ] **失败信息**：assert 添加 `msg` 参数解释为何失败
