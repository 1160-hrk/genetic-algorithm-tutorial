from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

"""
Demo (2‑D system) + 分布可視化
==============================
*GA の進化過程* を 20 世代ごとに XY 平面へ散布図として描画し、
タイル状 (subplot) に並べて確認するサンプルです。

実行方法
--------
    $ python examples/demo_root_2d.py

Matplotlib がインストールされていない場合は
    pip install matplotlib
で追加してください。
"""

import math
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from genalgo.population import Population
from genalgo.crossover import sbx
from genalgo.selection import tournament_select
from genalgo.mutation import gaussian

# ------------------------------------------------------------
# 方程式系（変更可）
# ------------------------------------------------------------

def equations(x: float, y: float) -> Tuple[float, float]:
    # f1 = x + y - 2**0.5      # 直線
    f1 = x + y - 1      # 直線
    f2 = x**2 + y**2 - 1.0         # 単位円
    return f1, f2


def fitness_fn(vec: np.ndarray) -> float:
    x, y = vec
    f1, f2 = equations(x, y)
    return f1 * f1 + f2 * f2

# ------------------------------------------------------------
# GA 設定
# ------------------------------------------------------------
POP_SIZE = 1000
GENERATIONS = 40
SEED = None
BOUNDS = ((-2.0, 2.0), (-2.0, 2.0))
# MUT_SIGMA = 0.25
MUT_SIGMA = 0.05
RECORD_EVERY = 1   # 何世代ごとに履歴を保存するか
PLOT_INTERVAL = 1  # 何世代間隔で表示するか
ENABLE_EARLY_STOP = True
# ------------------------------------------------------------
# GA 実行
# ------------------------------------------------------------

def run_ga() -> List[Tuple[int, np.ndarray]]:
    rng = np.random.default_rng(SEED)
    init = rng.uniform([b[0] for b in BOUNDS], [b[1] for b in BOUNDS], size=(POP_SIZE, 2))

    pop = Population(init_genes=init, fitness_fn=fitness_fn, rng=rng)
    pop.evolve(
        generations=GENERATIONS,
        record_every=RECORD_EVERY,
        enable_early_stop=ENABLE_EARLY_STOP,
        selector=lambda f, r: tournament_select(f, k=3, rng=r),
        crossover_op=lambda a, b, r: sbx(a, b, eta=1.0, rng=r),
        mutation_op=lambda x, r: gaussian(x, sigma=MUT_SIGMA, prob=1.0, bounds=BOUNDS, rng=r),
        bounds=BOUNDS,
        verbose=True,
    )
    return pop.gene_history

# ------------------------------------------------------------
# 可視化
# ------------------------------------------------------------

def plot_history(history: List[Tuple[int, np.ndarray]]) -> None:
    # 20 世代ごとに抽出
    frames = [(g, genes) for g, genes in history if g % PLOT_INTERVAL == 0]
    n_frames = len(frames)
    n_cols = math.ceil(math.sqrt(n_frames))
    n_rows = math.ceil(n_frames / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    for ax in axes.flat[n_frames:]:  # 余分な軸を非表示
        ax.axis("off")

    for idx, (gen, genes) in enumerate(frames):
        ax = axes.flat[idx]
        ax.scatter(genes[:, 0], genes[:, 1], s=10, alpha=0.6)
        ax.set_title(f"gen {gen}")
        ax.set_xlim(BOUNDS[0])
        ax.set_ylim(BOUNDS[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, lw=0.3)

    fig.suptitle("GA population snapshots (every 20 generations)", y=1.0)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
if __name__ == "__main__":
    history = run_ga()
    plot_history(history)
