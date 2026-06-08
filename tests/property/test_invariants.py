"""属性测试（Property-based Testing，hypothesis 驱动）。

与黑/白盒不同，属性测试不针对具体输入，而是对**对所有合法输入都成立的不变量**做
随机搜索式验证。hypothesis 会自动生成 100+ 组输入；遇到失败时自动 shrink 到最小反例。

编号规范：PROP-<不变量缩写>-<NN>
不变量清单：
  PROP-DET    确定性：相同输入两次 run 输出严格相等（NFR-02）
  PROP-MTM    MTM 守恒：MTM 后总价值 = MTM 前总价值 × (price_curr / price_prev)
  PROP-WC     WeightConfig.normalized 总和恒为 1（任意正权重）
  PROP-EWMA   ewma 单调性：常数序列 → 常数；λ=0.5 时输出介于首尾元素之间
  PROP-RANK   ScoringEngine 排序：top 池 score 不小于任何其他池
  PROP-SLIP   Slippage 单调性：trade_size 翻倍 → slippage 至少翻倍（相同 tvl）
  PROP-MDD    max_drawdown ∈ [0, 1]，对任意非负 NAV 序列
"""
from __future__ import annotations

import os
from datetime import datetime
from decimal import Decimal
from typing import Dict

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from backtest.cost_model import FrictionEstimator
from data_model.asset import AssetSnapshot, EnvSnapshot, PoolMetrics
from data_model.loader import build_asset_snapshots
from data.sample_data import generate_sample_data
from report.metrics import max_drawdown
from strategy.gain_estimator import APYDeltaGainEstimator
from strategy.interfaces import (
    Position,
    ScoringContext,
    ScoringParams,
    WeightConfig,
)
from strategy.scorers._common import zscore_dict
from strategy.scorers.cara import CARAUtilityAdjuster
from strategy.scorers.momentum import MomentumScorer, ewma
from strategy.scorers.risk_penalty import (
    DownsideVolPenaltyScorer,
    MaxDrawdownPenaltyScorer,
    TokenPriceMDDPenaltyScorer,
    TokenPriceVolPenaltyScorer,
)
from strategy.scoring_engine import ScoringEngine

pytestmark = [pytest.mark.property]


# 默认 settings：开发期可用 50 examples 跑得快；CI 应用 200+
_SETTINGS_FAST = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)


# =====================================================================
# PROP-DET：BacktestEngine 复现性
# =====================================================================

class TestDeterminism:

    @given(seed=st.integers(min_value=1, max_value=10000))
    @_SETTINGS_FAST
    def test_PROP_DET_01_same_synthetic_seed_same_nav(self, seed):
        """相同 seed 两次生成数据 + 两次 run → NAV 序列严格相等。"""
        from backtest.engine import BacktestEngine
        from strategy.reinvest_engine import ReinvestEngine
        from strategy.rotation_engine import RotationEngine

        pool_df, gas_df = generate_sample_data(seed=seed, n_days=60)
        snaps = build_asset_snapshots(pool_df, gas_df, config={"momentum_window": 14})

        def _engine():
            friction = FrictionEstimator()
            gain = APYDeltaGainEstimator(use_price_drift=False)
            return BacktestEngine(
                initial_capital=Decimal("100000"),
                scoring_engine=ScoringEngine(
                    params=ScoringParams(),
                    weight_cfg=WeightConfig({
                        "momentum": Decimal("0.4"), "vol_penalty": Decimal("0.2"),
                        "mdd_penalty": Decimal("0.2"), "cara": Decimal("0.2"),
                    }),
                    scorers=[
                        MomentumScorer(), DownsideVolPenaltyScorer(),
                        MaxDrawdownPenaltyScorer(), CARAUtilityAdjuster(),
                    ],
                ),
                rotation_engine=RotationEngine(
                    tau_reset=Decimal("0.05"), threshold=Decimal("0.001"),
                    gain_estimator=gain, friction_estimator=friction,
                ),
                reinvest_engine=ReinvestEngine(
                    friction_estimator=friction, gain_estimator=gain,
                    reinvest_window=30, risk_premium_multiplier=Decimal("1.5"),
                ),
            )

        r1 = _engine().run(snaps)
        r2 = _engine().run(snaps)
        assert r1.nav_log["nav"].equals(r2.nav_log["nav"])
        assert r1.trade_log.equals(r2.trade_log)


# =====================================================================
# PROP-WC：WeightConfig 归一化
# =====================================================================

class TestWeightConfigInvariants:

    @given(
        weights=st.lists(
            st.decimals(min_value=Decimal("0.001"), max_value=Decimal("100"),
                        allow_nan=False, allow_infinity=False),
            min_size=1, max_size=10,
        )
    )
    @_SETTINGS_FAST
    def test_PROP_WC_01_normalized_sums_to_one(self, weights):
        """任意正权重 dict → normalized 总和 = 1。"""
        d = {f"w_{i}": w for i, w in enumerate(weights)}
        n = WeightConfig(d).normalized()
        total = sum(n.weights.values())
        # 允许 Decimal 精度噪音 < 1e-20
        assert abs(total - Decimal(1)) < Decimal("1e-20")

    @given(
        weights=st.lists(
            st.decimals(min_value=Decimal("0.001"), max_value=Decimal("100"),
                        allow_nan=False, allow_infinity=False),
            min_size=1, max_size=10,
        )
    )
    @_SETTINGS_FAST
    def test_PROP_WC_02_normalized_preserves_ratios(self, weights):
        """归一化保留权重相对比例。"""
        d = {f"w_{i}": w for i, w in enumerate(weights)}
        n = WeightConfig(d).normalized()
        if len(weights) >= 2 and weights[0] > 0:
            ratio_orig = weights[1] / weights[0]
            ratio_norm = n.weights["w_1"] / n.weights["w_0"]
            # 允许浮点 / Decimal 精度差
            assert abs(ratio_norm - ratio_orig) < Decimal("1e-20")


# =====================================================================
# PROP-EWMA：EWMA 不变量
# =====================================================================

class TestEwmaInvariants:

    @given(
        x=st.decimals(min_value=Decimal("-1"), max_value=Decimal("1"),
                      allow_nan=False, allow_infinity=False, places=4),
        n=st.integers(min_value=1, max_value=50),
        lam_int=st.integers(min_value=1, max_value=99),
    )
    @_SETTINGS_FAST
    def test_PROP_EWMA_01_constant_series_returns_constant(self, x, n, lam_int):
        """任意 λ 下，恒定序列 → ewma 输出恒等于该常数。"""
        lam = Decimal(lam_int) / Decimal(100)
        out = ewma((x,) * n, lam)
        assert out == x

    @given(
        a=st.decimals(min_value=Decimal("0"), max_value=Decimal("1"),
                      allow_nan=False, allow_infinity=False, places=4),
        b=st.decimals(min_value=Decimal("0"), max_value=Decimal("1"),
                      allow_nan=False, allow_infinity=False, places=4),
    )
    @_SETTINGS_FAST
    def test_PROP_EWMA_02_two_element_output_between_inputs(self, a, b):
        """2 元素序列 (a, b)，λ ∈ (0,1) → ewma ∈ [min(a,b), max(a,b)]。"""
        out = ewma((a, b), Decimal("0.5"))
        assert min(a, b) <= out <= max(a, b)


# =====================================================================
# PROP-SLIP：滑点单调性 + 比例性
# =====================================================================

class TestSlippageInvariants:

    @given(
        trade_int=st.integers(min_value=1, max_value=10**8),
        tvl_int=st.integers(min_value=10**6, max_value=10**12),
    )
    @_SETTINGS_FAST
    def test_PROP_SLIP_01_doubling_trade_at_least_doubles_cost(
        self, trade_int, tvl_int
    ):
        """trade_size 翻倍 → slippage 成本至少翻倍（同档或升档）。"""
        fe = FrictionEstimator()
        t = Decimal(trade_int)
        tvl = Decimal(tvl_int)
        s1 = fe.estimate_slippage(t, tvl)
        s2 = fe.estimate_slippage(t * 2, tvl)
        # rate 至少不下降 → cost 至少翻倍
        assert s2 >= s1 * 2

    @given(
        trade_int=st.integers(min_value=1, max_value=10**8),
        tvl_int=st.integers(min_value=10**6, max_value=10**12),
    )
    @_SETTINGS_FAST
    def test_PROP_SLIP_02_cost_is_nonnegative(self, trade_int, tvl_int):
        fe = FrictionEstimator()
        cost = fe.estimate_slippage(Decimal(trade_int), Decimal(tvl_int))
        assert cost >= Decimal(0)


# =====================================================================
# PROP-MDD：max_drawdown 取值范围
# =====================================================================

class TestMaxDrawdownInvariants:

    @given(
        navs=st.lists(
            st.floats(min_value=0.0, max_value=1e10,
                      allow_nan=False, allow_infinity=False),
            min_size=1, max_size=200,
        )
    )
    @_SETTINGS_FAST
    def test_PROP_MDD_01_in_zero_one_range(self, navs):
        """任意非负 NAV 序列：MDD ∈ [0, 1]。"""
        out = max_drawdown(pd.Series(navs))
        assert Decimal(0) <= out <= Decimal(1)

    @given(
        start=st.floats(min_value=1.0, max_value=1e6,
                        allow_nan=False, allow_infinity=False),
        n=st.integers(min_value=1, max_value=100),
        step=st.floats(min_value=0.001, max_value=10.0,
                       allow_nan=False, allow_infinity=False),
    )
    @_SETTINGS_FAST
    def test_PROP_MDD_02_monotone_increasing_is_zero(self, start, n, step):
        """严格单调升 NAV → MDD = 0。"""
        navs = [start + step * i for i in range(n)]
        assert max_drawdown(pd.Series(navs)) == Decimal(0)


# =====================================================================
# PROP-RANK：排序 invariant
# =====================================================================

class TestRankingInvariants:

    @given(
        seed=st.integers(min_value=1, max_value=10000),
    )
    @_SETTINGS_FAST
    def test_PROP_RANK_01_top_score_is_max(self, seed):
        """top 池的综合得分应不低于其他池。"""
        np.random.seed(seed)
        pools = {}
        for i, pid in enumerate(["a", "b", "c"]):
            apy = float(np.random.uniform(0.01, 0.20))
            apy_series = tuple(
                Decimal(str(apy + np.random.normal(0, 0.005)))
                for _ in range(14)
            )
            pools[pid] = PoolMetrics(
                pool_id=pid, apy_series=apy_series,
                tvl=Decimal("100000000"), vol_30d=Decimal("0.02"),
                token_price=Decimal("1.0"), gas_base_fee=Decimal("0.0000001"),
            )
        env = EnvSnapshot(
            tick=0, timestamp=datetime(2024, 1, 1),
            oracle_price={pid: Decimal("1.0") for pid in pools},
            gas_base_fee=Decimal("0.0000001"),
            gas_priority_fee=Decimal("0.00000005"),
        )
        snap = AssetSnapshot(tick=0, pools=pools, env=env)

        engine = ScoringEngine(
            params=ScoringParams(),
            weight_cfg=WeightConfig({"momentum": Decimal("1.0")}),
            scorers=[MomentumScorer()],
        )
        ranking = engine.run(snap)
        scores = [p.score for p in ranking.rankings]
        assert scores[0] == max(scores)
        # 排序应稳定且降序
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# =====================================================================
# PROP-ZSC：z-score 性质
# =====================================================================

class TestZScoreInvariants:

    @given(
        values=st.lists(
            st.decimals(min_value=Decimal("-1000"), max_value=Decimal("1000"),
                        allow_nan=False, allow_infinity=False, places=4),
            min_size=2, max_size=20,
        )
    )
    @_SETTINGS_FAST
    def test_PROP_ZSC_01_zscore_sum_approximately_zero(self, values):
        """z-score 输出总和应 ≈ 0（数学性质）。"""
        # 避免全部相同（std=0 → 全 0）
        assume(len(set(values)) >= 2)
        d = {f"k_{i}": v for i, v in enumerate(values)}
        z = zscore_dict(d)
        total = sum(z.values())
        # 浮点 + Decimal 中转可能引入 1e-9 量级误差
        assert abs(total) < Decimal("1e-6")
