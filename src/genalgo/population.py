"""
Pluggable GA engine (selection / crossover / mutation areすべて関数注入型)
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Protocol, Sequence, Tuple

from .selection import tournament_select
from .crossover import one_point
from .mutation import gaussian

Array = np.ndarray
Bounds = Tuple[float, float] | Sequence[Tuple[float, float]] | None
RNG = np.random.Generator

# -- operator protocols -------------------------------------------------
class SelectOp(Protocol):
    def __call__(self, fitness: Array, rng: RNG) -> int: ...

class CrossOp(Protocol):
    def __call__(self, p1: Array, p2: Array, rng: RNG) -> Tuple[Array, Array]: ...

class MutOp(Protocol):
    def __call__(self, x: Array, rng: RNG) -> Array: ...


def _noop_crossover(a: Array, b: Array, rng: RNG) -> Tuple[Array, Array]:
    return a.copy(), b.copy()

# ----------------------------------------------------------------------
class Population:
    """Mini GA engine ・型安全・演算子差し替え自由"""

    def __init__(
        self,
        init_genes: Array,
        fitness_fn: Callable[[Array], float],
        *,
        rng: RNG | None = None,
    ):
        self.rng: RNG = rng or np.random.default_rng()
        self.genes = init_genes.copy()
        self.fitness_fn = fitness_fn
        self.fitness = np.apply_along_axis(fitness_fn, 1, self.genes)

    # ------------------------- utility --------------------------------
    def best(self) -> Tuple[Array, float]:
        idx = int(np.argmin(self.fitness))
        return self.genes[idx], float(self.fitness[idx])

    # ---------------------- one generation ----------------------------
    def _next_generation(
        self,
        *,
        selector: SelectOp,
        crossover_op: CrossOp,
        mutation_op: MutOp,
        crossover_rate: float,
    ):
        pop_n, _ = self.genes.shape
        new_genes = np.empty_like(self.genes)

        # エリート保持
        new_genes[0] = self.best()[0]

        i = 1
        while i < pop_n:
            p1 = self.genes[selector(self.fitness, self.rng)]
            p2 = self.genes[selector(self.fitness, self.rng)]

            c1, c2 = (
                crossover_op(p1, p2, self.rng)
                if self.rng.random() < crossover_rate
                else (p1.copy(), p2.copy())
            )

            new_genes[i] = mutation_op(c1, self.rng)
            if i + 1 < pop_n:
                new_genes[i + 1] = mutation_op(c2, self.rng)
            i += 2

        self.genes = new_genes
        self.fitness = np.apply_along_axis(self.fitness_fn, 1, self.genes)

    # --------------------------- public -------------------------------
    def evolve(
        self,
        generations: int,
        *,
        selector: SelectOp = tournament_select,
        crossover_op: CrossOp | None = None,
        mutation_op: MutOp | None = None,
        crossover_rate: float = 0.9,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
        bounds: Bounds = None,
        patience: int = 100,
        tol: float = 1e-8,
        target_fit: float | None = None,
        verbose: bool = False,
    ) -> Tuple[Array, float]:
        # デフォルト演算子
        if crossover_op is None:
            crossover_op = (
                _noop_crossover
                if self.genes.shape[1] < 2
                else lambda a, b, r: one_point(a, b, rng=r)
            )

        if mutation_op is None:
            def mutation_op(x: Array, rng: RNG) -> Array:  # type: ignore[override]
                return gaussian(x, sigma=mutation_sigma, prob=mutation_prob, bounds=bounds, rng=rng)

        best_prev = self.best()[1]
        stagnate = 0

        for g in range(generations):
            self._next_generation(
                selector=selector,
                crossover_op=crossover_op,
                mutation_op=mutation_op,
                crossover_rate=crossover_rate,
            )

            best_now = self.best()[1]
            if verbose and (g % 10 == 0 or g == generations - 1):
                print(f"[gen {g:4d}] best fitness = {best_now:.4g}")

            if target_fit is not None and best_now <= target_fit:
                break

            if abs(best_prev - best_now) < tol:
                stagnate += 1
                if stagnate >= patience:
                    break
            else:
                stagnate = 0

            best_prev = best_now

        return self.best()
