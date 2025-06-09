"""demo_root_1d_plot.py — 1 次元 GA の進化を可視化
=================================================

* g(x) = cos x − x の曲線と、各世代の個体を散布図で描画。
* `USE_INIT_GENES` を True/False で切り替え、
  - **True**  : 初期集団を自分で作って `init_genes` で渡す
  - **False** : `n_individuals` / `dim` / `bounds` で自動生成させる

"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# --- ローカル import 用パス設定 ----------------------------------------
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "../src")))

from genalgo.population import Population
from genalgo.selection import tournament_select
from genalgo.crossover import sbx
from genalgo.mutation import gaussian
from rootfinder.fitness import make_abs_fitness

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
USE_INIT_GENES = True      # False にすると自動生成モード
POP_SIZE = 100
GENERATIONS = 10
MUT_SIGMA = 0.01
MUT_PROB = 1
SNAPSHOT_EVERY = 1        # プロット間隔（世代数）
SEED = None
BOUNDS = (0.0, 2.0)        # 探索区間 [lo, hi]

fitness_fn = make_abs_fitness(lambda x: math.cos(x) - x)

# ----------------------------------------------------------------------
# GA 実行
# ----------------------------------------------------------------------

def run_ga() -> List[Tuple[int, np.ndarray]]:
    rng = np.random.default_rng(SEED)

    if USE_INIT_GENES:
        init = rng.uniform(BOUNDS[0], BOUNDS[1], size=(POP_SIZE, 1))
        pop = Population(init_genes=init, fitness_fn=fitness_fn, rng=rng)
    else:
        pop = Population(
            n_individuals=POP_SIZE,
            dim=1,
            bounds=BOUNDS,
            seed=SEED,
            fitness_fn=fitness_fn,
        )

    pop.evolve(
        generations=GENERATIONS,
        selector=lambda f, r: tournament_select(f, k=3, rng=r),
        crossover_op=lambda a, b, r: sbx(a, b, eta=0.6, rng=r),
        mutation_op=lambda x, r: gaussian(x, sigma=MUT_SIGMA, prob=MUT_PROB, bounds=BOUNDS, rng=r),
        record_every=SNAPSHOT_EVERY,
        verbose=False,
    )
    return pop.gene_history


# ----------------------------------------------------------------------
# 可視化
# ----------------------------------------------------------------------

def plot_snapshots(history: List[Tuple[int, np.ndarray]]):
    frames = [(g, genes) for g, genes in history if g % SNAPSHOT_EVERY == 0]
    n_frames = len(frames)
    n_cols = math.ceil(math.sqrt(n_frames))
    n_rows = math.ceil(n_frames / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    xs_curve = np.linspace(*BOUNDS, 400)
    ys_curve = np.cos(xs_curve) - xs_curve

    for i, (gen, genes) in enumerate(frames):
        ax = axes.flat[i]
        ax.plot(xs_curve, ys_curve, lw=1)
        x_pts = genes[:, 0]
        y_pts = np.cos(x_pts) - x_pts
        fitness_vals = np.abs(y_pts)
        best_fit = fitness_vals.min()
        ax.scatter(x_pts, y_pts, s=15, alpha=0.5, ec='tab:blue', fc="tab:blue")
        ax.hlines(0, 0, 2, linestyles=':', colors='k', alpha=0.5)
        ax.set_title(f"gen {gen} | best |f|={best_fit:.2e}")
        ax.grid(True, lw=0.3)
        ax.set_xlim(*BOUNDS)
        ax.set_ylim(min(ys_curve) - 0.2, max(ys_curve) + 0.2)
        if i % n_cols == 0:
            ax.set_ylabel("y")
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("x")

    for ax in axes.flat[n_frames:]:
        ax.axis("off")

    fig.suptitle(f"GA snapshots every {SNAPSHOT_EVERY} generations", y=0.97)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


if __name__ == "__main__":
    hist = run_ga()
    plot_snapshots(hist)
