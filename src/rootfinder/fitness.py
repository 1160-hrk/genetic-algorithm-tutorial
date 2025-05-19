import numpy as np
from typing import Callable

def make_abs_fitness(func: Callable[[float], float]) -> Callable[[np.ndarray], float]:
    """|f(x)| を最小化する適応度関数を生成"""
    def _fitness(x: np.ndarray) -> float:
        return abs(func(x[0]))
    return _fitness
