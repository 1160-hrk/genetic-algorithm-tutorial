"""demo_root_1d_sa.py — 自己適応 GA で 1 次元方程式 f(x)=0 を解く

方程式
    g(x) = cos x - x = 0      (根 ≈ 0.739085...)

特徴
* `Population(self_adaptive=True)` を利用
* 個体 = [x, log_sigma] の 2 次元ベクトル
* 収束ログと最終結果を表示

実行方法
---------
$ python examples/demo_root_1d_sa.py
"""
from __future__ import annotations
import sys
import os

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../src")))

import math
import numpy as np

from genalgo.population import Population
from genalgo import selectors as sel
from genalgo.crossover import sbx
from rootfinder.fitness import make_abs_fitness

# ------------------ 設定 ------------------
POP_SIZE    = 80
GENERATIONS = 400
SEED        = 0
BOUNDS      = (0.0, 2.0)   # gene 部分の範囲
INIT_SIGMA  = 0.01           # 初期 σ (自己適応用)
SIGMA_FLOOR = 1e-7   # 下限
TAU_SCALE   = 0.3   # τ を半分に

# 目的関数 |f(x)|
fitness_fn = make_abs_fitness(lambda x: math.cos(x) - x)

# ------------------ 初期集団 ------------------
rng = np.random.default_rng(SEED)
# gene 部分 (x)
genes = rng.uniform(BOUNDS[0], BOUNDS[1], size=(POP_SIZE, 1))
# log_sigma 部分 (初期 σ = INIT_SIGMA)
logs = np.full((POP_SIZE, 1), math.log(INIT_SIGMA))
init_genes = np.hstack([genes, logs])

# ------------------ Population ------------------
pop = Population(
    init_genes=init_genes,
    fitness_fn=fitness_fn,
    self_adaptive=True,      # ← ここがポイント
    bounds=BOUNDS,
    init_sigma=INIT_SIGMA,   # σ 初期値（sigmas 配列も生成される）
    sigma_floor=SIGMA_FLOOR,
    tau_scale=TAU_SCALE,
    seed=SEED,
)

best_gene, best_fit = pop.evolve(
    generations=GENERATIONS,
    selector=sel.tournament(k=5),
    crossover_op=lambda a, b, r: sbx(a, b, eta=1.0, rng=r),
    verbose=True,
)

x_best = best_gene[0]
print("\n=== Result (Self‑Adaptive GA) ===")
print(f"x ≃ {x_best:.10f}")
print(f"|f(x)| = {best_fit:.3e}")
print(f"σ (final) ≃ {pop.sigmas[np.argmin(pop.fitness)]:.3e}")
