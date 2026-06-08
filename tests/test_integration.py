"""Phase 6 端到端集成回归。

直接复用 run_example.run() 函数（不调 CLI），在 tmp_path 中走完
CSV 写入 → 加载 → 回测 → 报表 → 落盘 全链路。

核心断言（覆盖 NFR-01/02/04 与命令 6-2 的验收口径）：
  - 365 tick 处理完毕，运行时长 < 5s
  - nav_log/trade_log/score_log/reinvest_log 落盘 4 个 Parquet
  - GAS_SPIKE 在 nav_log 的 env_gas_base_fee 上正确传播（5×）
  - Pool_Exploit 后被攻击池的 total_score 显著下跌
  - LVR 由 oracle 与 token_price 偏离驱动，必须 > 0
  - 两次 run() 产物 NAV / trade / score 完全一致（NFR-02）
"""
from __future__ import annotations

import importlib
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

import run_example


@pytest.fixture
def isolated_run_example(tmp_path, monkeypatch):
    """把 run_example 的 DATA_DIR / CSV / OUTPUT 路径都重定向到 tmp_path，
    避免污染仓库目录。

    返回重定向后的 run_example 模块（注意：模块单例，测试结束后自动恢复 monkeypatch）。
    """
    data_dir = tmp_path / "data"
    output_dir = data_dir / "output"
    pool_csv = data_dir / "pools_sample.csv"
    gas_csv = data_dir / "gas_sample.csv"

    monkeypatch.setattr(run_example, "DATA_DIR", data_dir)
    monkeypatch.setattr(run_example, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(run_example, "POOL_CSV", pool_csv)
    monkeypatch.setattr(run_example, "GAS_CSV", gas_csv)
    return run_example


# =================================================================
# 主流程
# =================================================================

def test_run_example_full_pipeline(isolated_run_example):
    re = isolated_run_example
    result, metrics, attribution, dt = re.run(events_enabled=True, regen=True)

    # 365 tick 全部处理
    assert result.snapshots_processed == 365
    assert len(result.nav_log) == 365

    # NFR-04 性能：单次 365 tick × 3 池在合理硬件下 < 5s
    assert dt < 5.0, f"runtime {dt:.2f}s exceeds 5s budget"

    # 4 个 parquet 都落盘
    out = re.OUTPUT_DIR
    for fname in ("nav_log.parquet", "trade_log.parquet",
                  "reinvest_log.parquet", "score_log.parquet"):
        assert (out / fname).exists(), f"missing {fname}"

    # CSV 也写出来了
    assert (out / "metrics_report.csv").exists()
    assert (out / "attribution_report.csv").exists()


def test_gas_spike_propagates_5x(isolated_run_example):
    re = isolated_run_example
    result, _, _, _ = re.run(events_enabled=True, regen=True)

    # CSV 中 tick 150~154 的 base_fee × 5
    base_outside = float(
        result.nav_log.loc[result.nav_log["tick"] == 100, "env_gas_base_fee"].iloc[0]
    )
    base_inside = float(
        result.nav_log.loc[result.nav_log["tick"] == 152, "env_gas_base_fee"].iloc[0]
    )
    assert base_inside / base_outside == pytest.approx(5.0, rel=0.01)


def test_pool_exploit_drops_target_pool_score(isolated_run_example):
    re = isolated_run_example
    result, _, _, _ = re.run(events_enabled=True, regen=True)

    score_log = result.score_log
    s_before = score_log[
        (score_log["tick"] == 199) & (score_log["pool_id"] == "Curve_3Pool")
    ]["total_score"].iloc[0]
    s_after = score_log[
        (score_log["tick"] == 200) & (score_log["pool_id"] == "Curve_3Pool")
    ]["total_score"].iloc[0]
    assert s_after < s_before


def test_lvr_nonzero_from_oracle_divergence(isolated_run_example):
    re = isolated_run_example
    _, metrics, attribution, _ = re.run(events_enabled=True, regen=True)
    # oracle_price 的随机游走保证 LVR 非零
    assert metrics.total_lvr_cost > Decimal(0)
    assert attribution.lvr_pct > Decimal(0)


def test_attribution_decomposition_conserved(isolated_run_example):
    """理论 = 实际 + gas + slippage + lvr + idle，误差 ≤ 1（计价本位单位）。"""
    re = isolated_run_example
    _, _, attribution, _ = re.run(events_enabled=True, regen=True)
    reconstructed = (attribution.actual_return + attribution.gas_cost
                     + attribution.slippage_cost + attribution.lvr_cost
                     + attribution.rotation_idle_cost)
    diff = abs(reconstructed - attribution.theoretical_total_return)
    assert diff <= Decimal(1), f"decomposition gap {diff}"


# =================================================================
# 复现性（NFR-02）
# =================================================================

def test_two_runs_produce_identical_outputs(isolated_run_example):
    re = isolated_run_example
    r1, m1, a1, _ = re.run(events_enabled=True, regen=True)
    r2, m2, a2, _ = re.run(events_enabled=True, regen=False)   # 用同一 CSV

    assert r1.nav_log.equals(r2.nav_log)
    assert r1.trade_log.equals(r2.trade_log)
    assert r1.score_log.equals(r2.score_log)
    assert m1.annualized_return == m2.annualized_return
    assert m1.total_lvr_cost == m2.total_lvr_cost
    assert a1.theoretical_total_return == a2.theoretical_total_return
    assert a1.actual_return == a2.actual_return


# =================================================================
# 无事件路径 + verify CLI flag
# =================================================================

def test_no_events_path_smokes(isolated_run_example):
    re = isolated_run_example
    result, _, _, _ = re.run(events_enabled=False, regen=True)
    assert result.snapshots_processed == 365


def test_main_cli_runs_with_quiet_and_no_verify(isolated_run_example, capsys):
    re = isolated_run_example
    # main() 应当无异常退出；--no-verify 跳过断言路径
    re.main(["--quiet", "--no-verify", "--regen"])
    captured = capsys.readouterr()
    assert "snapshots from CSV" in captured.out
    assert "Annualized return" in captured.out
