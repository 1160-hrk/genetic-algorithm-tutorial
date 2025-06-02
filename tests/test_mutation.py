import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
from genalgo.mutation import gaussian

def test_gaussian_mutation_occurs():
    rng = np.random.default_rng(0)
    x = np.zeros(100)
    y = gaussian(x, sigma=1.0, prob=1.0, rng=rng)  # 必ず変異
    # 少なくとも一遺伝子は非ゼロ
    assert np.any(y != 0)

def test_bounds_clip():
    rng = np.random.default_rng(1)
    x = np.array([9.5])
    y = gaussian(x, sigma=10, prob=1.0, bounds=(-5, 5), rng=rng)
    assert -5 <= y[0] <= 5
