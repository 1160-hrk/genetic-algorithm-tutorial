# src/genalgo/mutation.py
# ==============================================================
# ガウス変異演算子
#
# - gaussian    : 固定 σ のガウス変異
# - gaussian_sa : 自己適応 σ を遺伝子に含めて進化させる新バージョン
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
    """配列 *x* を bounds 内に収める（bounds=None なら無変更）。"""
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
    各遺伝子を確率 *prob* で `N(0, σ²)` ノイズ加算。

    Parameters
    ----------
    x : ndarray
        親個体 (1-D)。
    sigma : float
        ノイズの標準偏差。
    prob : float
        変異確率 (per gene)。
    bounds : tuple | list | None
        変異後にクリップする範囲。None なら無制限。
    rng : numpy.random.Generator, optional
        再現用 RNG。
    """
    if rng is None:
        rng = np.random.default_rng()

    child = x.copy()
    mask = rng.random(child.shape) < prob
    child[mask] += rng.normal(0.0, sigma, size=mask.sum())
    return _clip(child, bounds)


# ------------------------------------------------------------------
# 2. 自己適応 σ ガウス変異
# ------------------------------------------------------------------
def gaussian_sa(
    indiv: Array,
    *,
    tau: float | None = None,
    bounds: Bounds = None,
    rng: RNG | None = None,
) -> Array:
    """
    自己適応バージョン。

    個体を `[x0, x1, …, x_{d−1}, log_sigma]` とし、
    最後の要素が σ の自然対数を保持すると仮定。

    アルゴリズム
    -----------
    1. `log_sigma' = log_sigma + τ * N(0,1)`
    2. `σ' = exp(log_sigma')`
    3. `x_i' = x_i + σ' * N(0,1)`  (i=0…d−1, prob=1)

    Parameters
    ----------
    tau : float, optional
        σ の進化強度 (既定 1/√d)。
    bounds : tuple | list | None
        遺伝子部分のクリップ範囲 (log_sigma はクリップしない)。
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = indiv.size - 1
    if dim < 1:
        raise ValueError("individual must contain at least one gene plus log_sigma")

    if tau is None:
        tau = 1.0 / np.sqrt(dim)

    x = indiv[:-1]
    log_sigma = indiv[-1]

    # σ の進化
    log_sigma_new = log_sigma + tau * rng.normal()
    sigma_new = np.exp(log_sigma_new)

    # 遺伝子のガウス変異 (prob=1)
    child_genes = x + sigma_new * rng.normal(size=dim)
    child_genes = _clip(child_genes, bounds)

    child = np.empty_like(indiv)
    child[:-1] = child_genes
    child[-1] = log_sigma_new
    return child

