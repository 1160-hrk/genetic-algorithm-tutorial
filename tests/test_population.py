# tests/test_population.py
# ==============================================================
# Population クラスの基本動作テスト
#   1. 固定 σ モードで収束するか
#   2. 自己適応 σ モードで sigmas が更新されるか
#   3. gene_history が record_every 間隔で保存されるか
# ==============================================================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import math

from genalgo.population import Population
from genalgo.selection import tournament_select
from genalgo.crossover import sbx
from genalgo.mutation import gaussian
from rootfinder.fitness import make_abs_fitness


# -----------------------------------------------------------
# 1. 固定 σ モードで収束確認
# -----------------------------------------------------------
def test_population_fixed_sigma_converges():
    f = lambda x: math.cos(x) - x
    fitness = make_abs_fitness(f)

    pop = Population(
        n_individuals=60,
        dim=1,
        bounds=(0.0, 2.0),
        seed=0,
        fitness_fn=fitness,
        self_adaptive=False,
    )

    _, best_fit = pop.evolve(
        generations=300,
        selector=lambda fit, r: tournament_select(fit, k=3, rng=r),
        crossover_op=lambda a, b, r: sbx(a, b, eta=0.6, rng=r),
        mutation_op=lambda x, r: gaussian(x, sigma=0.1, prob=1.0, bounds=(0, 2), rng=r),
        verbose=False,
    )

    # 10^-4 程度に収束しているか
    assert best_fit < 1e-4


# -----------------------------------------------------------
# 2. 自己適応 σ が更新されるか
# -----------------------------------------------------------
def test_population_self_adaptive_sigma_changes():
    f = lambda x: math.cos(x) - x
    fitness = make_abs_fitness(f)

    rng = np.random.default_rng(1)
    init_genes = rng.uniform(0, 2, size=(30, 1))  # 1 遺伝子のみ
    pop = Population(
        init_genes=init_genes,
        fitness_fn=fitness,
        self_adaptive=True,
        init_sigma=0.2,
        bounds=(0, 2),
        rng=rng,
    )

    sig_before = pop.sigmas.copy()
    pop.evolve(generations=1, verbose=False)   # 1 世代だけ回す
    sig_after = pop.sigmas

    # σ 配列のどこかが変わっているはず
    assert not np.allclose(sig_before, sig_after)


# -----------------------------------------------------------
# 3. gene_history が正しく保存されるか
# -----------------------------------------------------------
def test_gene_history_record_every():
    f = lambda x: (x - 1.0) ** 2
    fitness = make_abs_fitness(f)

    pop = Population(
        fitness_fn=fitness,
        n_individuals=20,
        dim=1,
        bounds=(0, 2),
        seed=2,
    )

    record_every = 5
    gens = 12
    pop.evolve(
        generations=gens,
        record_every=record_every,
        verbose=False,
    )

    recorded_gens = [g for g, _ in pop.gene_history]
    # 初期 (0) と 5,10 世代が保存される (11,12 はしきい値未満)
    assert recorded_gens == [0, 5, 10]
