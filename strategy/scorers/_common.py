"""Scorer 共用工具：z-score 归一化、Decimal/numpy 桥接等。"""
from __future__ import annotations

from decimal import Decimal
from typing import Dict, Mapping

import numpy as np


def to_decimal(x) -> Decimal:
    """通过 str 中转，避免 float -> Decimal 精度漂移。"""
    if isinstance(x, Decimal):
        return x
    return Decimal(str(x))


def zscore_dict(raw: Mapping[str, Decimal]) -> Dict[str, Decimal]:
    """按 dict 的 values 做 z-score 归一化。

    全部相同（std=0）时返回全 0；空 dict 返回空 dict。
    使用 np.float64 计算，最终结果转 Decimal —— 中间统计精度足够，
    且确保跨平台、跨运行数值一致（NFR-02）。
    """
    if not raw:
        return {}
    keys = list(raw.keys())
    arr = np.array([float(raw[k]) for k in keys], dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std == 0.0:
        return {k: Decimal(0) for k in keys}
    z = (arr - mean) / std
    return {keys[i]: to_decimal(float(z[i])) for i in range(len(keys))}
