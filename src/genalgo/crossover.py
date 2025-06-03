"""
Crossover operators for real-valued genomes.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

Array = np.ndarray
RNG = np.random.Generator

# ----------------------------------------------------------------------
# 1. One-point crossover
# ----------------------------------------------------------------------
def one_point(a: Array, b: Array, rng: RNG | None = None) -> Tuple[Array, Array]:
    """親 2 個体から 1 点交叉で子を返す。1 遺伝子ならそのままコピー。"""
    if rng is None:
        rng = np.random.default_rng()
    if a.shape != b.shape:
        raise ValueError("Parents must have identical shape.")

    n = a.size
    if n < 2:  # 切れ目なし
        return a.copy(), b.copy()

    cut = rng.integers(1, n)  # 1 … n-1
    return (
        np.concatenate((a[:cut], b[cut:])).copy(),
        np.concatenate((b[:cut], a[cut:])).copy(),
    )


# ----------------------------------------------------------------------
# 2. Uniform crossover
# ----------------------------------------------------------------------
def uniform(a: Array, b: Array, p: float = 0.5, rng: RNG | None = None) -> Tuple[Array, Array]:
    """各遺伝子を確率 p で交換する一様交叉。"""
    if rng is None:
        rng = np.random.default_rng()
    if a.shape != b.shape:
        raise ValueError("Parents must have identical shape.")

    mask = rng.random(a.shape) < p
    c1, c2 = a.copy(), b.copy()
    c1[mask], c2[mask] = b[mask], a[mask]
    return c1, c2


# ----------------------------------------------------------------------
# 3. BLX-α (Blend Crossover)
# ----------------------------------------------------------------------
def blx_alpha(a: Array, b: Array, alpha: float = 0.5, rng: RNG | None = None) -> Tuple[Array, Array]:
    """親区間を α 倍拡張した一様分布から子を生成。"""
    if rng is None:
        rng = np.random.default_rng()
    if a.shape != b.shape:
        raise ValueError("Parents must have identical shape.")

    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    diff = hi - lo
    lower = lo - alpha * diff
    upper = hi + alpha * diff
    return rng.uniform(lower, upper), rng.uniform(lower, upper)


# ----------------------------------------------------------------------
# 4. SBX (Simulated Binary Crossover)
# ----------------------------------------------------------------------
def sbx(a: Array, b: Array, eta: float = 2.0, rng: RNG | None = None) -> Tuple[Array, Array]:
    """Deb & Agrawal (1995) の SBX。eta↑ で親に近い子を生成。"""
    if rng is None:
        rng = np.random.default_rng()
    if a.shape != b.shape:
        raise ValueError("Parents must have identical shape.")

    u = rng.random(a.shape)
    beta = np.where(
        u <= 0.5,
        (2 * u) ** (1.0 / (eta + 1)),
        (1 / (2 * (1 - u))) ** (1.0 / (eta + 1)),
    )
    c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    return c1, c2
