"""Phase 1 数据层单元测试。"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics, freeze_decimal_iter
from data_model.loader import build_asset_snapshots, load_gas_csv, load_pool_csv
from data_model.preprocessor import (
    align_timeseries,
    apply_capacity_decay,
    interpolate_missing,
    remove_outliers_iqr,
)


# ---------- 数据类不可变性 ----------

def _make_pool_metrics(pool_id: str = "pool_a") -> PoolMetrics:
    return PoolMetrics(
        pool_id=pool_id,
        apy_series=(Decimal("0.05"), Decimal("0.06")),
        tvl=Decimal("1000000"),
        vol_30d=Decimal("0.02"),
        token_price=Decimal("1.0"),
        gas_base_fee=Decimal("20"),
    )


def test_pool_metrics_is_frozen():
    pm = _make_pool_metrics()
    with pytest.raises(FrozenInstanceError):
        pm.tvl = Decimal("999")  # type: ignore[misc]


def test_pool_metrics_apy_series_coerced_to_tuple():
    pm = PoolMetrics(
        pool_id="x",
        apy_series=[Decimal("0.05"), Decimal("0.06")],  # type: ignore[arg-type]
        tvl=Decimal("1"),
        vol_30d=Decimal("0"),
        token_price=Decimal("1"),
        gas_base_fee=Decimal("0"),
    )
    assert isinstance(pm.apy_series, tuple)


def test_pool_metrics_rejects_non_decimal_money_field():
    with pytest.raises(TypeError):
        PoolMetrics(
            pool_id="x",
            apy_series=(Decimal("0.05"),),
            tvl=1.0,  # type: ignore[arg-type]
            vol_30d=Decimal("0"),
            token_price=Decimal("1"),
            gas_base_fee=Decimal("0"),
        )


def test_env_snapshot_oracle_price_is_immutable_view():
    src = {"pool_a": Decimal("1.0")}
    env = EnvSnapshot(
        tick=0,
        timestamp=pd.Timestamp("2024-01-01").to_pydatetime(),
        oracle_price=src,
        gas_base_fee=Decimal("20"),
        gas_priority_fee=Decimal("1"),
    )
    # 外部修改原 dict 不应影响 env（已经 dict() 拷贝）
    src["pool_a"] = Decimal("999")
    assert env.oracle_price["pool_a"] == Decimal("1.0")
    with pytest.raises(TypeError):
        env.oracle_price["pool_a"] = Decimal("2")  # type: ignore[index]


def test_asset_snapshot_pool_ids_is_sorted():
    p1 = _make_pool_metrics("zzz")
    p2 = _make_pool_metrics("aaa")
    env = EnvSnapshot(
        tick=0,
        timestamp=pd.Timestamp("2024-01-01").to_pydatetime(),
        oracle_price={},
        gas_base_fee=Decimal("0"),
        gas_priority_fee=Decimal("0"),
    )
    snap = AssetSnapshot(tick=0, pools={"zzz": p1, "aaa": p2}, env=env)
    assert snap.pool_ids() == ("aaa", "zzz")


def test_freeze_decimal_iter_handles_mixed_types():
    out = freeze_decimal_iter([1, "2.5", Decimal("3.14")])
    assert out == (Decimal(1), Decimal("2.5"), Decimal("3.14"))
    assert all(isinstance(x, Decimal) for x in out)


# ---------- CSV 加载 ----------

def _write_sample_csvs(tmp_path: Path) -> tuple[Path, Path]:
    pool_path = tmp_path / "pools.csv"
    gas_path = tmp_path / "gas.csv"
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    pool_rows = []
    for pid in ["pool_a", "pool_b"]:
        for i, ts in enumerate(dates):
            pool_rows.append({
                "timestamp": ts,
                "pool_id": pid,
                "apy": 0.05 + 0.001 * i,
                "tvl": 1_000_000 + 10_000 * i,
                "token_price": 1.0,
            })
    pd.DataFrame(pool_rows).to_csv(pool_path, index=False)

    gas_rows = [{"timestamp": ts, "base_fee": 20.0, "priority_fee": 1.5} for ts in dates]
    pd.DataFrame(gas_rows).to_csv(gas_path, index=False)
    return pool_path, gas_path


def test_load_pool_csv_validates_columns(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2024-01-01"], "pool_id": ["x"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="缺少必需列"):
        load_pool_csv(bad)


def test_build_asset_snapshots_smoke(tmp_path: Path):
    pool_path, gas_path = _write_sample_csvs(tmp_path)
    pool_df = load_pool_csv(pool_path)
    gas_df = load_gas_csv(gas_path)
    snaps = build_asset_snapshots(pool_df, gas_df, config={"momentum_window": 3})

    assert len(snaps) == 5
    assert all(isinstance(s, AssetSnapshot) for s in snaps)
    # 在第 4 个 tick (index 3)，回看窗口 3 应该有 3 项 APY
    assert len(snaps[3].pools["pool_a"].apy_series) == 3
    # 在第 0 个 tick，回看窗口 1 项
    assert len(snaps[0].pools["pool_a"].apy_series) == 1
    # tick 单调递增
    assert [s.tick for s in snaps] == list(range(5))


def test_build_asset_snapshots_is_deterministic(tmp_path: Path):
    """NFR-02：相同输入两次构建结果应完全一致。"""
    pool_path, gas_path = _write_sample_csvs(tmp_path)
    cfg = {"momentum_window": 3}

    snaps1 = build_asset_snapshots(load_pool_csv(pool_path), load_gas_csv(gas_path), cfg)
    snaps2 = build_asset_snapshots(load_pool_csv(pool_path), load_gas_csv(gas_path), cfg)

    assert len(snaps1) == len(snaps2)
    for a, b in zip(snaps1, snaps2):
        assert a.tick == b.tick
        assert a.pool_ids() == b.pool_ids()
        for pid in a.pool_ids():
            assert a.pools[pid] == b.pools[pid]


def test_build_asset_snapshots_uses_oracle_price_column_if_present(tmp_path: Path):
    """oracle_price 列存在时，env.oracle_price 应取该列；否则回退到 token_price。"""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    pool_rows = [
        {"timestamp": ts, "pool_id": "x", "apy": 0.05,
         "tvl": 1_000_000, "token_price": 1.0, "oracle_price": 1.005}
        for ts in dates
    ]
    pool_path = tmp_path / "pools.csv"
    pd.DataFrame(pool_rows).to_csv(pool_path, index=False)

    gas_path = tmp_path / "gas.csv"
    pd.DataFrame([{"timestamp": ts, "base_fee": 1e-7, "priority_fee": 5e-8} for ts in dates]).to_csv(gas_path, index=False)

    snaps = build_asset_snapshots(load_pool_csv(pool_path), load_gas_csv(gas_path), {"momentum_window": 3})
    # token_price 不变；oracle_price 应用 1.005
    assert snaps[0].pools["x"].token_price == Decimal("1.0")
    assert snaps[0].env.oracle_price["x"] == Decimal("1.005")


def test_build_asset_snapshots_falls_back_when_no_oracle_column(tmp_path: Path):
    """没有 oracle_price 列时 env.oracle_price 退化为 token_price（向后兼容）。"""
    pool_path, gas_path = _write_sample_csvs(tmp_path)   # 不含 oracle_price 列
    snaps = build_asset_snapshots(load_pool_csv(pool_path), load_gas_csv(gas_path), {"momentum_window": 3})
    for snap in snaps:
        for pid, pm in snap.pools.items():
            assert snap.env.oracle_price[pid] == pm.token_price


# ---------- 预处理 ----------

def test_align_timeseries_with_groups():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01", "2024-01-03",  # pool_a 缺 01-02
            "2024-01-01", "2024-01-02", "2024-01-03",
        ]),
        "pool_id": ["pool_a", "pool_a", "pool_b", "pool_b", "pool_b"],
        "apy": [0.05, 0.06, 0.04, 0.045, 0.05],
    })
    out = align_timeseries(df, freq="D", group_col="pool_id")
    pool_a = out[out["pool_id"] == "pool_a"]
    assert len(pool_a) == 3  # 01-02 被补齐为 NaN 行
    assert pool_a["apy"].isna().sum() == 1


def test_interpolate_missing_fills_nan():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
        "pool_id": ["pool_a"] * 5,
        "apy": [0.05, np.nan, np.nan, 0.08, 0.09],
    })
    out = interpolate_missing(df, ["apy"], group_col="pool_id")
    assert out["apy"].isna().sum() == 0
    # 线性插值：(0.05, x, y, 0.08) → x=0.06, y=0.07
    assert out.iloc[1]["apy"] == pytest.approx(0.06)
    assert out.iloc[2]["apy"] == pytest.approx(0.07)


def test_remove_outliers_iqr_replaces_extreme_values():
    base = [0.05, 0.052, 0.048, 0.051, 0.049, 0.05, 0.053, 0.05]
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(base) + 1, freq="D"),
        "pool_id": ["pool_a"] * (len(base) + 1),
        "apy": base + [99.0],   # 末尾极端异常值
    })
    out = remove_outliers_iqr(df, ["apy"], k=3.0, group_col="pool_id")
    assert out["apy"].max() < 1.0  # 异常值被替换
    assert out["apy"].isna().sum() == 0  # ffill/bfill 修复


# ---------- 容量衰减 ----------

def test_capacity_decay_dilutes_apy_when_capital_large():
    full = apply_capacity_decay(
        apy_nominal=Decimal("0.10"),
        tvl=Decimal("1000000"),
        capital=Decimal("0"),
    )
    diluted = apply_capacity_decay(
        apy_nominal=Decimal("0.10"),
        tvl=Decimal("1000000"),
        capital=Decimal("1000000"),  # 注入等量资金
    )
    assert full == Decimal("0.10")
    assert diluted == Decimal("0.05")  # 各占 50% 份额


def test_capacity_decay_lending_high_utilization_penalty():
    """利用率 > kink 时，等量注资的 APY 应低于 kink 以下场景。"""
    low_u = apply_capacity_decay(
        apy_nominal=Decimal("0.10"),
        tvl=Decimal("1000000"),
        capital=Decimal("100000"),
        pool_kind="lending",
        utilization=Decimal("0.5"),
    )
    high_u = apply_capacity_decay(
        apy_nominal=Decimal("0.10"),
        tvl=Decimal("1000000"),
        capital=Decimal("100000"),
        pool_kind="lending",
        utilization=Decimal("0.95"),
    )
    assert high_u < low_u


def test_capacity_decay_zero_pool_returns_zero():
    out = apply_capacity_decay(
        apy_nominal=Decimal("0.10"),
        tvl=Decimal("0"),
        capital=Decimal("0"),
    )
    assert out == Decimal(0)


def test_capacity_decay_returns_decimal():
    out = apply_capacity_decay(
        apy_nominal=0.10,  # type: ignore[arg-type]
        tvl=1_000_000,     # type: ignore[arg-type]
        capital=100_000,   # type: ignore[arg-type]
    )
    assert isinstance(out, Decimal)
