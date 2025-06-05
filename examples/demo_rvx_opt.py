from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import math
from typing import Tuple
import numpy as np

import rovibrational_excitation as rvx
from genalgo.population import Population
from genalgo.crossover import sbx
from genalgo.selection import tournament_select
from genalgo.mutation import gaussian

# ------------------------------------------------------------
# Define the system of equations
# ------------------------------------------------------------

V_max = 2
J_max = 1
basis = rvx.LinMolBasis(V_max, J_max, use_M=True)

def equations(x: float, y: float) -> Tuple[float, float]:
    """Return (f1, f2) for given (x, y). Modify as needed."""
    f1 = x + y - 2**0.5
    # f2 = x ** 2 + y ** 2 - 10.0
    f2 = x ** 2 + y ** 2 - 1
    return f1, f2


def fitness_fn(vec: np.ndarray) -> float:
    """Scalar fitness = sum of squares → 0 when both equations are satisfied."""
    x, y = vec
    f1, f2 = equations(x, y)
    return f1 * f1 + f2 * f2


# ------------------------------------------------------------
# GA CONFIG — tweak these values only
# ------------------------------------------------------------
POP_SIZE = 120       # 集団サイズ
GENERATIONS = 400    # 世代数
SEED = 0             # 乱数シード
BOUNDS = ((-5.0, 5.0), (-5.0, 5.0))  # 探索範囲 (x_min, x_max), (y_min, y_max)
MUTATION_SIGMA = 0.2 # ガウス変異の σ

# ------------------------------------------------------------
# GA execution
# ------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)
    init_genes = rng.uniform(
        [b[0] for b in BOUNDS], [b[1] for b in BOUNDS], size=(POP_SIZE, 2)
    )

    pop = Population(init_genes, fitness_fn, rng=rng)

    best_gene, best_fit = pop.evolve(
        generations=GENERATIONS,
        selector=lambda f, r: tournament_select(f, k=3, rng=r),
        crossover_op=lambda a, b, r: sbx(a, b, eta=1.0, rng=r),
        mutation_op=lambda x, r: gaussian(
            x, sigma=MUTATION_SIGMA, prob=1.0, bounds=BOUNDS, rng=r
        ),
        bounds=BOUNDS,
        verbose=True,
    )

    x, y = best_gene
    f1, f2 = equations(x, y)

    print("\nBest candidate found:")
    print(f"x = {x:.6f}, y = {y:.6f}")
    print(f"sum of squares = {best_fit:.3e}")
    print(f"f1 = {f1:.3e}, f2 = {f2:.3e}")


if __name__ == "__main__":
    main()
