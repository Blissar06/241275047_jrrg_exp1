"""Phase 7-B：DefiLlama fetcher 单元测试（全部用 mock，不发真实网络）。

策略：monkeypatch `urllib.request.urlopen` 返回 io.BytesIO，
封装期望的 JSON payload，覆盖：
  - list_top_pools 过滤逻辑与排序
  - fetch_pool_history 字段转换 + APY/100
  - build_pool_csv_from_defillama 多池合并
  - 缓存命中（第二次调用不再触发 urlopen）
  - 网络错误抛 OnchainFetchError
  - synthesize_gas_for_range 与 spike 注入
"""
from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

from data.onchain_fetcher import (
    OnchainFetchError,
    PoolMeta,
    build_pool_csv_from_defillama,
    fetch_pool_history,
    list_top_pools,
    synthesize_gas_for_range,
)


# =================================================================
# Mock helpers
# =================================================================

class _FakeResponse:
    def __init__(self, payload: dict):
        self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._buf.getvalue()


def _patch_urlopen(monkeypatch, route: dict):
    """route: {url_substring: payload}.  Url 命中 substring 时返回对应 payload。"""
    call_log: list[str] = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        call_log.append(url)
        for sub, payload in route.items():
            if sub in url:
                return _FakeResponse(payload)
        raise urllib.error.URLError(f"no mock for {url}")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return call_log


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """把默认缓存目录重定向到 tmp_path，避免污染仓库。"""
    cache_dir = tmp_path / "cache"
    return cache_dir


# =================================================================
# /pools 端点
# =================================================================

_POOLS_PAYLOAD = {
    "data": [
        {"pool": "uuid-1", "symbol": "USDC", "project": "aave-v3",
         "chain": "Ethereum", "tvlUsd": 50_000_000, "apy": 5.2},
        {"pool": "uuid-2", "symbol": "DAI", "project": "compound-v3",
         "chain": "Ethereum", "tvlUsd": 20_000_000, "apy": 3.5},
        {"pool": "uuid-3", "symbol": "USDT", "project": "aave-v3",
         "chain": "Arbitrum", "tvlUsd": 8_000_000, "apy": 4.0},
        {"pool": "uuid-4", "symbol": "low-tvl", "project": "x",
         "chain": "Ethereum", "tvlUsd": 100, "apy": 99.0},
    ],
}


def test_list_top_pools_filters_chain(isolated_cache, monkeypatch):
    _patch_urlopen(monkeypatch, {"/pools": _POOLS_PAYLOAD})
    out = list_top_pools(
        chain="Ethereum", min_tvl_usd=1_000_000,
        cache_dir=isolated_cache, use_cache=False,
    )
    chains = {p.chain for p in out}
    assert chains == {"Ethereum"}


def test_list_top_pools_filters_project(isolated_cache, monkeypatch):
    _patch_urlopen(monkeypatch, {"/pools": _POOLS_PAYLOAD})
    out = list_top_pools(
        chain=None, project="aave-v3", min_tvl_usd=1_000_000,
        cache_dir=isolated_cache, use_cache=False,
    )
    assert {p.project for p in out} == {"aave-v3"}


def test_list_top_pools_sorts_by_tvl_desc(isolated_cache, monkeypatch):
    _patch_urlopen(monkeypatch, {"/pools": _POOLS_PAYLOAD})
    out = list_top_pools(
        chain=None, min_tvl_usd=1_000_000,
        cache_dir=isolated_cache, use_cache=False,
    )
    tvls = [p.tvl_usd for p in out]
    assert tvls == sorted(tvls, reverse=True)


def test_list_top_pools_filters_min_tvl(isolated_cache, monkeypatch):
    _patch_urlopen(monkeypatch, {"/pools": _POOLS_PAYLOAD})
    out = list_top_pools(
        chain=None, min_tvl_usd=10_000_000,
        cache_dir=isolated_cache, use_cache=False,
    )
    assert all(p.tvl_usd >= 10_000_000 for p in out)
    assert len(out) == 2


def test_list_top_pools_uses_cache_on_second_call(isolated_cache, monkeypatch):
    call_log = _patch_urlopen(monkeypatch, {"/pools": _POOLS_PAYLOAD})
    list_top_pools(cache_dir=isolated_cache, use_cache=True)
    list_top_pools(cache_dir=isolated_cache, use_cache=True)
    # 第二次应命中缓存
    pools_calls = [u for u in call_log if "/pools" in u]
    assert len(pools_calls) == 1


def test_list_top_pools_network_error_propagates(isolated_cache, monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("DNS fail")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(OnchainFetchError):
        list_top_pools(cache_dir=isolated_cache, use_cache=False)


# =================================================================
# /chart/{pool_id} 端点
# =================================================================

_CHART_PAYLOAD = {
    "data": [
        {"timestamp": "2024-01-01T00:00:00.000Z", "apy": 5.0, "tvlUsd": 10_000_000},
        {"timestamp": "2024-01-02T00:00:00.000Z", "apy": 5.5, "tvlUsd": 10_100_000},
        {"timestamp": "2024-01-03T00:00:00.000Z", "apy": 4.8, "tvlUsd": 10_050_000},
    ],
}


def test_fetch_pool_history_converts_apy_to_decimal(isolated_cache, monkeypatch):
    _patch_urlopen(monkeypatch, {"/chart/": _CHART_PAYLOAD})
    df = fetch_pool_history("uuid-1", cache_dir=isolated_cache, use_cache=False)

    assert len(df) == 3
    # 5.0% → 0.05
    assert df["apy"].iloc[0] == pytest.approx(0.05)
    assert df["tvl"].iloc[0] == 10_000_000


def test_fetch_pool_history_timestamps_are_tz_naive(isolated_cache, monkeypatch):
    _patch_urlopen(monkeypatch, {"/chart/": _CHART_PAYLOAD})
    df = fetch_pool_history("uuid-1", cache_dir=isolated_cache, use_cache=False)
    assert df["timestamp"].dt.tz is None


def test_fetch_pool_history_empty_data_raises(isolated_cache, monkeypatch):
    _patch_urlopen(monkeypatch, {"/chart/": {"data": []}})
    with pytest.raises(OnchainFetchError):
        fetch_pool_history("uuid-x", cache_dir=isolated_cache, use_cache=False)


def test_fetch_pool_history_uses_cache_on_second_call(isolated_cache, monkeypatch):
    call_log = _patch_urlopen(monkeypatch, {"/chart/": _CHART_PAYLOAD})
    fetch_pool_history("uuid-1", cache_dir=isolated_cache, use_cache=True)
    fetch_pool_history("uuid-1", cache_dir=isolated_cache, use_cache=True)
    assert len([u for u in call_log if "/chart/" in u]) == 1


# =================================================================
# 批量构建项目 CSV
# =================================================================

def test_build_pool_csv_concatenates_multiple_pools(isolated_cache, monkeypatch, tmp_path):
    # 同时 mock /pools（用于 find_pool_meta_by_id）和 /chart
    _patch_urlopen(monkeypatch, {
        "/pools": _POOLS_PAYLOAD,
        "/chart/": _CHART_PAYLOAD,
    })
    out = tmp_path / "real_pools.csv"
    df = build_pool_csv_from_defillama(
        selections=[
            {"pool_id": "uuid-1", "display": "Aave-USDC"},
            {"pool_id": "uuid-2", "display": "Compound-DAI"},
        ],
        output_path=out, n_days=3,
        cache_dir=isolated_cache, use_cache=False,
    )

    assert out.exists()
    assert set(df["pool_id"]) == {"Aave-USDC", "Compound-DAI"}
    assert set(df.columns) == {
        "timestamp", "pool_id", "apy", "tvl", "token_price", "oracle_price",
    }
    # 2 池 × 3 行
    assert len(df) == 6
    # USDC 池 symbol 在 SYMBOL_TOKEN_ADDRESS 中标 None → token_price 应稳定在 1.0
    assert (df[df["pool_id"] == "Aave-USDC"]["token_price"] == 1.0).all()


def test_build_pool_csv_empty_selections_raises(isolated_cache):
    with pytest.raises(ValueError):
        build_pool_csv_from_defillama(selections=[], cache_dir=isolated_cache)


def test_build_pool_csv_truncates_to_n_days(isolated_cache, monkeypatch):
    _patch_urlopen(monkeypatch, {
        "/pools": _POOLS_PAYLOAD,
        "/chart/": _CHART_PAYLOAD,
    })
    df = build_pool_csv_from_defillama(
        selections=[{"pool_id": "uuid-1", "display": "X"}],
        n_days=2,
        cache_dir=isolated_cache, use_cache=False,
    )
    # 3 行 chart → 取末尾 2 行
    assert len(df) == 2


# =================================================================
# 合成 Gas
# =================================================================

def test_synthesize_gas_constant_by_default():
    ts = pd.date_range("2024-01-01", periods=5, freq="D")
    df = synthesize_gas_for_range(ts)
    assert len(df) == 5
    assert df["base_fee"].nunique() == 1


def test_synthesize_gas_applies_spike_windows():
    ts = pd.date_range("2024-01-01", periods=10, freq="D")
    df = synthesize_gas_for_range(
        ts, base_fee=1.0, priority_fee=0.5,
        spike_windows=[(3, 2, 5.0)],
    )
    assert df["base_fee"].iloc[2] == 1.0
    assert df["base_fee"].iloc[3] == 5.0
    assert df["base_fee"].iloc[4] == 5.0
    assert df["base_fee"].iloc[5] == 1.0


def test_synthesize_gas_writes_csv(tmp_path):
    ts = pd.date_range("2024-01-01", periods=3, freq="D")
    out = tmp_path / "gas.csv"
    df = synthesize_gas_for_range(ts, output_path=out)
    assert out.exists()
    read_back = pd.read_csv(out)
    assert len(read_back) == 3


# =================================================================
# PoolMeta 值对象
# =================================================================

def test_pool_meta_is_frozen():
    meta = PoolMeta(
        pool_id="x", symbol="USDC", project="aave",
        chain="Ethereum", tvl_usd=1.0, apy=5.0,
    )
    with pytest.raises(Exception):
        meta.tvl_usd = 999.0  # type: ignore[misc]
