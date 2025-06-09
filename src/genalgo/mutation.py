# src/genalgo/mutation.py
# ==============================================================
# 変異演算子
#
# - gaussian(x, …)          : 固定 σ のガウス変異（個体 1 本を返す）
# - gaussian_sa(idx, pop, …) : 自己適応 σ で in-place 変異（返り値なし）
#
# Population.self_adaptive == True のときに
#   mutation_sa(i, pop, rng) と呼び出すことを想定。
# ==============================================================

from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple

Array = np.ndarray
Bounds = Tuple[float, float] | Sequence[Tuple[float, float]] | None
RNG = np.random.Generator

# ------------------------------------------------------------------
# 内部: クリップ関数
# ------------------------------------------------------------------
def _clip(x: Array, bounds: Bounds) -> Array:
    """配列 *x* を bounds 内に収めて返す。bounds=None なら無変更。"""
    if bounds is None:
        return x
    if isinstance(bounds[0], (int, float)):
        lo, hi = bounds  # type: ignore[misc]
        return np.clip(x, lo, hi)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return np.clip(x, lo, hi)


# ------------------------------------------------------------------
# 1. 固定 σ ガウス変異
# ------------------------------------------------------------------
def gaussian(
    x: Array,
    *,
    sigma: float = 0.1,
    prob: float = 0.1,
    bounds: Bounds = None,
    rng: RNG | None = None,
) -> Array:
    """
    各遺伝子を確率 *prob* で `N(0, σ²)` ノイズ加算し、クリップして返す。
    """
    if rng is None:
        rng = np.random.default_rng()

    child = x.copy()
    mask = rng.random(child.shape) < prob
    child[mask] += rng.normal(0.0, sigma, size=mask.sum())
    return _clip(child, bounds)


# ------------------------------------------------------------------
# 2. 自己適応 σ ガウス変異（Population 依存）
# ------------------------------------------------------------------
def gaussian_sa(
    idx: int,
    pop: "Population",
    *,
    tau_scale: float = 1.0,
    rng: RNG | None = None,
) -> None:
    """
    個体 `idx` を **in-place** で自己適応ガウス変異する。

    前提
    ----
    * `pop.genes[idx]` … 遺伝子ベクトル (shape = (dim,))   ← 直接更新
    * `pop.sigmas[idx]` … 実数 σ                           ← 直接更新
    * `pop.bounds`      … クリップ範囲 (Population が保持)

    アルゴリズム
    ------------
    σ ← σ * exp( τ * N(0,1) )      # σ の自己適応
    genes ← genes + σ * N(0,1)^dim # 全遺伝子を変異 (prob=1)
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = pop.genes.shape[1]
    tau = tau_scale / np.sqrt(dim)

    # 1) σ の進化
    pop.sigmas[idx] *= np.exp(tau * rng.normal())

    # 2) 遺伝子変異
    pop.genes[idx] += pop.sigmas[idx] * rng.normal(size=dim)
    pop.genes[idx] = _clip(pop.genes[idx], pop.bounds)
