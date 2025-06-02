"""
Crossover operators for real-valued genomes.

使い方
------
child1, child2 = one_point(parent1, parent2, rng)
child1, child2 = uniform(parent1, parent2, p=0.5, rng=rng)
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

Array = np.ndarray
RNG = np.random.Generator


# ----------------------------------------------------------------------
# 1. One-point crossover
# ----------------------------------------------------------------------
def one_point(
    a: Array,
    b: Array,
    rng: RNG | None = None,
) -> Tuple[Array, Array]:
    """
    Split parents at a random position and swap the tail parts.

    Returns two new children (copies).  Parents are **not** modified.
    """
    if rng is None:
        rng = np.random.default_rng()

    if a.shape != b.shape:
        raise ValueError("Parents must have identical shape")

    n = a.size
    if n < 2:                     # ← 1 次元遺伝子は切れないので何もしない
        return a.copy(), b.copy()

    cut = rng.integers(1, n)          # 1 … n-1
    child1 = np.concatenate((a[:cut], b[cut:])).copy()
    child2 = np.concatenate((b[:cut], a[cut:])).copy()
    return child1, child2


# ----------------------------------------------------------------------
# 2. Uniform crossover (gene-wise shuffle)
# ----------------------------------------------------------------------
def uniform(
    a: Array,
    b: Array,
    p: float = 0.5,
    rng: RNG | None = None,
) -> Tuple[Array, Array]:
    """
    For each gene, swap with probability *p* (default 0.5).
    """
    if rng is None:
        rng = np.random.default_rng()

    if a.shape != b.shape:
        raise ValueError("Parents must have identical shape")

    mask = rng.random(a.shape) < p
    child1, child2 = a.copy(), b.copy()
    child1[mask], child2[mask] = b[mask], a[mask]
    return child1, child2
