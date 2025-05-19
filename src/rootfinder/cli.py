#!/usr/bin/env python
import argparse, math
from .search import find_root

def main():
    ap = argparse.ArgumentParser(description="GA で f(x)=0 の根を探す")
    ap.add_argument("--func", required=True,
                    help='Python 式 (e.g. "math.sin(x)-0.5")')
    args = ap.parse_args()

    expr = args.func
    func = lambda x: eval(expr, {"math": math, "x": x})
    root, cost = find_root(func)
    print(f"Approx. root: x = {root[0]:.6g}, |f(x)| = {cost:.3e}")

if __name__ == "__main__":
    main()
