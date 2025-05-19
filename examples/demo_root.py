"""
使い方デモ:  cos(x) - x = 0  (有名な固有値 ≈ 0.739085...)
"""
import math
from rootfinder.search import find_root

if __name__ == "__main__":
    root, cost = find_root(lambda x: math.cos(x) - x,
                           pop_size=150, generations=500,
                           x_min=0.0, x_max=2.0)
    print("root ≈", root[0], "  |f| =", cost)
