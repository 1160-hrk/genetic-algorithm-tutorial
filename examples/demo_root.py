import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import math, numpy as np
from genalgo.population import Population
from genalgo.crossover import sbx
from genalgo.selection import rank_select
from genalgo.selection import tournament_select
from rootfinder.fitness import make_abs_fitness

f = lambda x: math.cos(x) - x
fitness = make_abs_fitness(f)
init = np.random.default_rng(0).uniform(0, 2, (80, 1))

pop = Population(init, fitness)
best_gene, best_fit = pop.evolve(
    generations=600,
    selector=lambda f,r: tournament_select(f, k=3, rng=r),
    crossover_op=lambda a,b,r: sbx(a, b, eta=0.6, rng=r),
    mutation_prob=1.0,
    mutation_sigma=0.15,
    verbose=True,
)

print(best_gene[0], best_fit)
