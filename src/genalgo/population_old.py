# src/genalgo/population.py
# ==============================================================
# A small yet robust GA engine with pluggable operators
# ==============================================================

from __future__ import annotations

import numpy as np
from typing import Callable, Sequence, Tuple, Protocol

from .selection import tournament_select
from .crossover import one_point
from .mutation import gaussian

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Array = np.ndarray
Bounds = Tuple[float, float] | Sequence[Tuple[float, float]] | None
RNG = np.random.Generator


class CrossOp(Protocol):
    """crossover(parent1, parent2, rng) -> (child1, child2)."""

    def __call__(
        self, parent1: Array, parent2: Array, rng: RNG
    ) -> Tuple[Array, Array]: ...


class MutOp(Protocol):
    """mutation(individual, rng) -> mutated individual."""

    def __call__(self, x: Array, rng: RNG) -> Array: ...


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _noop_crossover(a: Array, b: Array, rng: RNG) -> Tuple[Array, Array]:
    """Return deep copies of parents (for 1-gene genomes etc.)."""
    return a.copy(), b.copy()


def _clip(x: Array, bounds: Bounds) -> Array:
    if bounds is None:
        return x
    lo, hi = (
        bounds
        if isinstance(bounds[0], (int, float))
        else (np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds]))
    )
    return np.clip(x, lo, hi)


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
class Population:
    """
    Genetic-Algorithm mini engine with pluggable operators.

    Parameters
    ----------
    init_genes : (N, dim) ndarray
        Initial population (each row = genome).
    fitness_fn : Callable[[ndarray], float]
        Genome → scalar fitness (lower is better).
    rng : np.random.Generator, optional
        RNG; default = np.random.default_rng().
    """

    def __init__(
        self,
        init_genes: Array,
        fitness_fn: Callable[[Array], float],
        *,
        rng: RNG | None = None,
    ) -> None:
        self.rng: RNG = rng or np.random.default_rng()
        self.genes: Array = init_genes.copy()
        self.fitness_fn: Callable[[Array], float] = fitness_fn
        self.fitness: Array = np.apply_along_axis(fitness_fn, 1, self.genes)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def best(self) -> Tuple[Array, float]:
        idx = int(np.argmin(self.fitness))
        return self.genes[idx], float(self.fitness[idx])

    # ------------------------------------------------------------------
    # One generation
    # ------------------------------------------------------------------
    def _next_generation(
        self,
        *,
        tournament_k: int,
        crossover_rate: float,
        crossover_op: CrossOp,
        mutation_op: MutOp,
    ) -> None:
        pop_size, _ = self.genes.shape
        new_genes = np.empty_like(self.genes)

        # --- elitism ---------------------------------------------------
        elite_gene, elite_fit = self.best()
        new_genes[0] = elite_gene

        # --- offspring production -------------------------------------
        i = 1
        while i < pop_size:
            # parent selection
            p1 = self.genes[
                tournament_select(self.fitness, k=tournament_k, rng=self.rng)
            ]
            p2 = self.genes[
                tournament_select(self.fitness, k=tournament_k, rng=self.rng)
            ]

            # crossover
            if self.rng.random() < crossover_rate:
                c1, c2 = crossover_op(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # mutation
            c1 = mutation_op(c1, self.rng)
            c2 = mutation_op(c2, self.rng)

            # store
            new_genes[i] = c1
            if i + 1 < pop_size:
                new_genes[i + 1] = c2
            i += 2

        # evaluate new population
        self.genes = new_genes
        self.fitness = np.apply_along_axis(self.fitness_fn, 1, self.genes)

        # make sure elite survives (safety net)
        worst_idx = int(np.argmax(self.fitness))
        if self.fitness[worst_idx] > elite_fit:
            self.genes[worst_idx] = elite_gene
            self.fitness[worst_idx] = elite_fit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evolve(
        self,
        generations: int,
        *,
        tournament_k: int = 3,
        crossover_rate: float = 0.9,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
        bounds: Bounds = None,
        crossover_op: CrossOp | None = None,
        mutation_op: MutOp | None = None,
        target_fit: float | None = None,
        patience: int = 100,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> Tuple[Array, float]:
        """
        Run evolution; returns (best_gene, best_fitness).

        Early stopping:
            * `target_fit`  到達で停止
            * `patience`    世代連続で改善幅 < `tol` なら停止
        """
        # ---- default operators ---------------------------------------
        if crossover_op is None:
            # auto choose: 1-gene ⇒ noop, otherwise one_point
            crossover_op = (
                _noop_crossover
                if self.genes.shape[1] < 2
                else lambda a, b, rng: one_point(a, b, rng=rng)
            )

        if mutation_op is None:

            def mutation_op(x: Array, rng: RNG) -> Array:  # type: ignore[override]
                return gaussian(
                    x,
                    sigma=mutation_sigma,
                    prob=mutation_prob,
                    bounds=bounds,
                    rng=rng,
                )

        # ---- evolution loop ------------------------------------------
        best_fit_prev = self.best()[1]
        stagnate = 0

        for g in range(generations):
            self._next_generation(
                tournament_k=tournament_k,
                crossover_rate=crossover_rate,
                crossover_op=crossover_op,
                mutation_op=mutation_op,
            )

            best_fit_now = self.best()[1]

            # verbose
            if verbose and (g % 10 == 0 or g == generations - 1):
                print(f"[gen {g:4d}] best fitness = {best_fit_now:.4g}")

            # early-stop: target fitness
            if target_fit is not None and best_fit_now <= target_fit:
                if verbose:
                    print(f"Target fitness reached (≤ {target_fit}).")
                break

            # early-stop: stagnation
            if abs(best_fit_prev - best_fit_now) < tol:
                stagnate += 1
                if stagnate >= patience:
                    if verbose:
                        print("No significant improvement; stopping early.")
                    break
            else:
                stagnate = 0

            best_fit_prev = best_fit_now

        return self.best()
