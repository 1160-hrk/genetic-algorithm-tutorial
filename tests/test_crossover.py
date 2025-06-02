import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import numpy as np
from genalgo.crossover import one_point, uniform

def test_one_point_length():
    a = np.array([0, 0, 0, 0])
    b = np.array([1, 1, 1, 1])
    child1, child2 = one_point(a, b, rng=np.random.default_rng(0))
    assert child1.size == a.size and child2.size == a.size
    # 子は親遺伝子の再配置のみ
    assert set(child1) <= {0, 1} and set(child2) <= {0, 1}

def test_uniform_swap_ratio():
    rng = np.random.default_rng(42)
    a, b = np.zeros(1000), np.ones(1000)
    c1, c2 = uniform(a, b, p=0.3, rng=rng)
    swap_rate = c1.sum() / 1000  # 1 の割合
    assert 0.25 < swap_rate < 0.35   # おおよそ 30 %
