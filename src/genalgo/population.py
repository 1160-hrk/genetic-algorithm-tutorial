from __future__ import annotations
import numpy as np
from typing import Callable

class Population:
    """遺伝子集団を管理し、評価関数を保持する簡易クラス (雛形)"""
    def __init__(self, init_genes: np.ndarray, fitness_fn: Callable[[np.ndarray], float]):
        self.genes = init_genes      # (N, dim)
        self.fitness_fn = fitness_fn # scalar fitness
        self.fitness = np.apply_along_axis(fitness_fn, 1, init_genes)

    def best(self) -> tuple[np.ndarray, float]:
        idx = np.argmin(self.fitness)
        return self.genes[idx], self.fitness[idx]
