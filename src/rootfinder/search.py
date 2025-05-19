import numpy as np
from genalgo.population import Population
from .fitness import make_abs_fitness

def find_root(func, *,
              pop_size: int = 100,
              generations: int = 300,
              x_min: float = -10.0,
              x_max: float = 10.0):
    rng = np.random.default_rng()
    init_genes = rng.uniform(x_min, x_max, size=(pop_size, 1))
    pop = Population(init_genes, make_abs_fitness(func))

    # --- 超簡易 GA ループ (デモ用) ---
    for _ in range(generations):
        # ① 全個体を乱数で微小変異
        children = pop.genes + rng.normal(0, 0.1, size=pop.genes.shape)
        # ② 評価して親子で良い方を残す
        child_fit = np.apply_along_axis(pop.fitness_fn, 1, children)
        better = child_fit < pop.fitness
        pop.genes[better] = children[better]
        pop.fitness[better] = child_fit[better]

    return pop.best()
