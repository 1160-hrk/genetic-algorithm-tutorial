"""
selectors.py — selection ラッパ集
--------------------------------
Population.evolve() が期待する
    selector(fitness: ndarray, rng: Generator) -> int
の形に包み直すヘルパを提供します。
"""

from typing import Callable
import numpy as np

# 元の選択関数を import
from .selection import (
    tournament_select,
    roulette_select,
    rank_select,
)

# 型エイリアス
Array = np.ndarray
RNG   = np.random.Generator
Selector = Callable[[Array, RNG], int]

# -------- factory wrappers ---------
def tournament(k: int = 3) -> Selector:
    """k トーナメント選択のラッパ"""
    return lambda fit, rng: tournament_select(fit, k=k, rng=rng)

def roulette() -> Selector:
    """ルーレット選択のラッパ"""
    return lambda fit, rng: roulette_select(fit, rng=rng)

def rank(selective_pressure: float = 1.7) -> Selector:
    """ランク選択のラッパ"""
    return lambda fit, rng: rank_select(fit, rng=rng, selective_pressure=selective_pressure)
