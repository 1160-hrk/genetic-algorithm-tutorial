"""Selection operators for the genetic algorithm core.

公開関数
--------
tournament_select(pop_fitness, k, rng) -> np.ndarray
    k 個の個体をランダムに抽出し、その中で最も適応度が良いものを返す。
roulette_select(pop_fitness, n_select, rng) -> np.ndarray
    適応度に比例した確率で n_select 個体を選ぶ（重複あり）。
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
    k: int,
    rng: RNG | None = None,
) -> int:
    """
    Return the index of the best individual among *k* randomly sampled ones.

    Parameters
    ----------
    fitness : (N,) array_like
        Lower fitness is better (minimization).  If高い方が良い場合は符号を変えて呼び出してください。
    k : int
        Tournament size (k >= 2 recommended).
    rng : np.random.Generator, optional
        Random‐number generator; default = np.random.default_rng().

    Returns
    -------
    int
        Index of the selected individual in the population.
    """
    if rng is None:
        rng = np.random.default_rng()

    pop_size = fitness.shape[0]
    indices = rng.integers(0, pop_size, size=k)
    best_idx = indices[np.argmin(fitness[indices])]
    return int(best_idx)


# ----------------------------------------------------------------------
# 2. Roulette-wheel (fitness-proportional) selection
# ----------------------------------------------------------------------
def roulette_select(
    fitness: Array,
    n_select: int,
    rng: RNG | None = None,
    transform: Callable[[Array], Array] | None = None,
) -> Array:
    """
    Select *n_select* indices with probability ∝ transformed fitness.

    Parameters
    ----------
    fitness : (N,) array_like
        Lower fitness is better.  Internally (1 / fitness) をデフォルト変換に使用。
    n_select : int
        Number of individuals to select (with replacement).
    rng : np.random.Generator, optional
        RNG; default = np.random.default_rng().
    transform : Callable, optional
        Function mapping fitness → non-negative weight.
        None の場合は `lambda f: 1 / (f + ε)` を使う。

    Returns
    -------
    np.ndarray
        Selected indices, shape = (n_select,)
    """
    if rng is None:
        rng = np.random.default_rng()

    eps = np.finfo(float).eps
    if transform is None:
        weights = 1.0 / (fitness + eps)
    else:
        weights = transform(fitness)

    weights_sum = np.sum(weights)
    if weights_sum <= 0:
        raise ValueError("All selection weights are zero or negative.")

    probs = weights / weights_sum
    return rng.choice(len(fitness), size=n_select, replace=True, p=probs)
