import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
from genalgo.selection import tournament_select, roulette_select

def test_tournament_select():
    fit = np.array([3.0, 1.0, 2.0])  # best = index 1
    rng = np.random.default_rng(42)
    # k = 3 なら必ず最良を選ぶ
    assert tournament_select(fit, k=3, rng=rng) == 1

def test_roulette_select_bias():
    fit = np.array([10.0, 1.0])  # 1.0 の方が良い
    rng = np.random.default_rng(0)
    sel = roulette_select(fit, n_select=1000, rng=rng)
    # index 1 が 80% 以上選ばれるはず（確率的検査）
    assert np.mean(sel == 1) > 0.8
