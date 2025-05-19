import math, numpy as np
from rootfinder.search import find_root

def test_simple_root():
    f = lambda x: x**2 - 4
    root, cost = find_root(f, generations=50, pop_size=30, x_min=-5, x_max=5)
    assert abs(root[0] - 2.0) < 1e-1 and cost < 1e-2
