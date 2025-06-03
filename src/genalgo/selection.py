"""
Selection operators for the genetic-algorithm core (real-valued genomes).

公開関数
--------
tournament_select(fitness, k, rng) -> int
    k 個取り出して最良 1 個体のインデックスを返す。
roulette_select(fitness, rng, transform) -> int
    適応度に比例した確率で 1 個体を選ぶ。
rank_select(fitness, rng, selective_pressure) -> int
    線形ランク選択（Baker 1985）。
"""

from __future__ import annotations

import numpy as np
from typing import Callable

Array = np.ndarray
RNG = np.random.Generator

# ----------------------------------------------------------------------
# 1. Tournament selection
# ----------------------------------------------------------------------
def tournament_select(
    fitness: Array,
    k: int = 3,
    rng: RNG | None = None,
) -> int:
    """k-tournament で 1 個体のインデックスを返す（minimisation 前提）。"""
    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, len(fitness), size=k)
    return int(idx[np.argmin(fitness[idx])])


# ----------------------------------------------------------------------
# 2. Roulette-wheel selection (fitness-proportional)
# ----------------------------------------------------------------------
def roulette_select(
    fitness: Array,
    rng: RNG | None = None,
    transform: Callable[[Array], Array] | None = None,
) -> int:
    """適応度に反比例した重みで 1 個体を選択。"""
    if rng is None:
        rng = np.random.default_rng()

    eps = np.finfo(float).eps
    weights = 1.0 / (fitness + eps) if transform is None else transform(fitness)
    probs = weights / weights.sum()
    return int(rng.choice(len(fitness), p=probs))


# ----------------------------------------------------------------------
# 3. Rank-based selection (linear rank)
# ----------------------------------------------------------------------
def rank_select(
    fitness: Array,
    rng: RNG | None = None,
    selective_pressure: float = 1.7,
) -> int:
    """
    線形ランク選択（Baker 1985）。
    selective_pressure : 1.0–2.0（大きいほど選択圧が強い）
    """
    if rng is None:
        rng = np.random.default_rng()

    ranks = np.argsort(np.argsort(fitness))  # 0 が最良
    n = len(fitness)
    s = selective_pressure
    probs = (2 - s) / n + 2 * ranks * (s - 1) / (n * (n - 1))
    probs /= probs.sum()
    return int(rng.choice(n, p=probs))
