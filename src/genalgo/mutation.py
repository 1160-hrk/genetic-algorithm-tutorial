"""
Mutation operators for real-valued genomes.

例
---
mut = gaussian(individual, sigma=0.1, prob=0.2, rng=rng, bounds=(-5, 5))
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Sequence

Array = np.ndarray
Bounds = Tuple[float, float] | Sequence[Tuple[float, float]]
RNG = np.random.Generator


def _clip(x: Array, bounds: Bounds | None) -> Array:
    if bounds is None:
        return x
    lo, hi = (
        bounds if isinstance(bounds[0], (int, float)) else zip(*bounds)  # per-gene 可
    )
    return np.clip(x, lo, hi)


def gaussian(
    x: Array,
    *,
    sigma: float = 0.1,
    prob: float = 0.1,
    bounds: Bounds | None = None,
    rng: RNG | None = None,
) -> Array:
    """
    Return a **new** individual with Gaussian noise added to each gene
    with probability *prob* (per gene).

    Parameters
    ----------
    x : np.ndarray
        1-D genome (will be copied; original remains intact).
    sigma : float
        Standard deviation of the added noise.
    prob : float
        Mutation probability per gene (0–1).
    bounds : tuple or list of tuples, optional
        (lo, hi) global bounds or per-gene bounds.  If None, no clipping.
    """
    if rng is None:
        rng = np.random.default_rng()

    child = x.copy()
    mask = rng.random(child.shape) < prob
    child[mask] += rng.normal(0.0, sigma, size=mask.sum())
    return _clip(child, bounds)
