# tests/test_mutation.py
# ===========================================================
# 固定 σ / 自己適応 σ 変異のテスト（修正版）
#   * 自己適応テストで clip=0 になるケースを回避するため
#     初期遺伝子を 1.0 に変更
# ===========================================================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import math
import numpy as np

from genalgo.mutation import gaussian, gaussian_sa
from genalgo.population import Population
from rootfinder.fitness import make_abs_fitness


# -----------------------------------------------------------
# 1. 固定 σ ガウス変異
# -----------------------------------------------------------
def test_gaussian_basic():
    rng = np.random.default_rng(0)
    x = np.zeros(5)
    child = gaussian(x, sigma=0.5, prob=1.0, bounds=(-1, 1), rng=rng)

    # 変異している
    assert not np.allclose(child, x)
    # 範囲内
    assert np.all(child >= -1) and np.all(child <= 1)


# -----------------------------------------------------------
# 2. 自己適応 σ ガウス変異 (in-place)
# -----------------------------------------------------------
def test_gaussian_sa_inplace():
    rng = np.random.default_rng(0)

    # 初期遺伝子を 1.0 として clip で元に戻るケースを避ける
    init_genes = np.full((2, 1), 1.0)

    pop = Population(
        init_genes=init_genes,
        fitness_fn=make_abs_fitness(lambda x: math.cos(x) - x),
        self_adaptive=True,
        init_sigma=0.1,
        bounds=(0, 2),
        rng=rng,
    )

    genes_before  = pop.genes.copy()
    sigmas_before = pop.sigmas.copy()

    # 個体 0 を自己適応変異
    gaussian_sa(0, pop, rng=rng)

    # genes または sigmas が確実に変化していることを確認
    assert not np.allclose(pop.sigmas[0], sigmas_before[0]), "sigma did not change"
    assert not np.allclose(pop.genes[0],  genes_before[0]),  "genes did not change"

    # 変異していない個体 1 は変化しない
    assert np.allclose(pop.genes[1], genes_before[1])
    assert math.isclose(pop.sigmas[1], sigmas_before[1])

    # クリップ範囲内
    assert 0.0 <= pop.genes[0, 0] <= 2.0
