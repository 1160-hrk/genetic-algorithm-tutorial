"""
使い方デモ:  cos(x) - x = 0  (有名な固有値 ≈ 0.739085...)
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import math, numpy as np
from genalgo.population import Population
from rootfinder.fitness import make_abs_fitness

if __name__ == "__main__":
    f = lambda x: math.cos(x) - x
    fitness = make_abs_fitness(f)

    rng = np.random.default_rng(0)
    init = rng.uniform(0, 2, size=(100, 1))

    pop = Population(init, fitness, rng=rng)
    best_gene, best_fit = pop.evolve(
        generations=300,
        tournament_k=3,
        crossover_rate=0.9,
        mutation_prob=0.2,
        mutation_sigma=0.05,
        bounds=(0, 2),
        verbose=True,
    )
    print("\nResult:", best_gene[0], " |f| =", best_fit)
