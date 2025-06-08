"""population.py  —  汎用 GA エンジン（履歴記録機能付き）
====================================================
このモジュールは **演算子（選択・交叉・変異）を関数として注入**できる
柔軟な GA コアを提供します。2025‑06 版では、

* 指定した間隔ごとに **集団（genes）を履歴として保存** する機能
* Python の型ヒント・Protocol による演算子シグネチャの保証
* 早期停止（改善停滞・目標適応度）

を備えています。

使い方（最小例）
-----------------
```python
pop = Population(init_genes, fitness_fn)
best_gene, best_fit = pop.evolve(
    generations=300,
    record_every=10,           # 10 世代ごとに履歴を記録
)
# 履歴は pop.gene_history で参照できる
for gen, genes in pop.gene_history:
    print(gen, genes.shape)    # (N, dim)
```

主な公開属性
-------------
* `genes`          : 最新世代の遺伝子行列 (N, dim)
* `fitness`        : 各個体のスカラー適応度 (N,)
* `gene_history`   : `List[Tuple[int, np.ndarray]]`  — (世代番号, genes のコピー)

演算子関数の引数シグネチャ
---------------------------
* **選択**   `selector(fitness: Array, rng) -> int`
* **交叉**   `crossover_op(p1: Array, p2: Array, rng) -> Tuple[Array, Array]`
* **変異**   `mutation_op(ind: Array, rng) -> Array`

これらさえ満たせば自由に差し替え可能です。

===========================================================

2025/06/08 改修
------------
* **初期化インターフェースを拡張**し、次の 2 通りを選択可能にした。

  1. **従来どおり** `init_genes` 行列を渡す。
  2. **形状 + 個体数 + 範囲** を渡してライブラリ側で一様乱数生成。

* 追加引数
  * `n_individuals : int | None` — 個体数
  * `dim          : int | None` — 遺伝子長
  * `bounds       : Bounds | None` — (lo, hi) か各遺伝子ごとのリスト
  * `seed         : int | None` — RNG シード（`rng` より優先）

使い分け例
-----------
```python
# 旧: 行列を自前で作る
init = np.random.rand(100, 3)
pop  = Population(init_genes=init, fitness_fn=f)

# 新: 行列を自動生成
pop = Population(n_individuals=100, dim=3, bounds=(0.0, 1.0), seed=42, fitness_fn=f)
```

注意
----
* `init_genes` が渡された場合はその他の生成パラメータは無視される。
* どちらも渡されなかった場合は `ValueError` を送出。
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, Tuple, List
import numpy as np
from .selection import tournament_select
from .crossover import one_point
from .mutation import gaussian

Array = np.ndarray
Bounds = Tuple[float, float] | Sequence[Tuple[float, float]] | None
RNG = np.random.Generator

# ---------------------- operator protocols ----------------------------
class SelectOp(Protocol):
    def __call__(self, fitness: Array, rng: RNG) -> int: ...

class CrossOp(Protocol):
    def __call__(self, p1: Array, p2: Array, rng: RNG) -> Tuple[Array, Array]: ...

class MutOp(Protocol):
    def __call__(self, ind: Array, rng: RNG) -> Array: ...


# 1 次元用交叉スキップ

def _noop_crossover(a: Array, b: Array, rng: RNG) -> Tuple[Array, Array]:
    return a.copy(), b.copy()


# ---------------------------- Population ------------------------------
class Population:
    """遺伝的アルゴリズム用の集団クラス。

    Parameters (どちらか必須)
    -------------------------
    init_genes : ndarray, optional
        `(N, dim)` 形状の初期集団行列。
    n_individuals : int, optional
        個体数。`init_genes` が無い場合は必須。
    dim : int, optional
        遺伝子長。`init_genes` が無い場合は必須。
    bounds : tuple | list, default (0.0, 1.0)
        初期乱数を生成する一様区間。グローバル `(lo, hi)` か
        `[(lo0, hi0), (lo1, hi1), ...]`。
    seed : int | None
        乱数シード。`rng` より優先。
    rng : numpy.random.Generator | None
        RNG インスタンス。未指定なら `seed` から生成。
    fitness_fn : Callable[[ndarray], float]
        個体 → スカラー適応度関数（小さいほど良い）。
    """

    def __init__(
        self,
        *,
        fitness_fn: Callable[[Array], float],
        init_genes: Array | None = None,
        n_individuals: int | None = None,
        dim: int | None = None,
        bounds: Bounds = (0.0, 1.0),
        seed: int | None = None,
        rng: RNG | None = None,
    ) -> None:
        # RNG 準備
        if rng is None:
            rng = np.random.default_rng(seed)
        self.rng: RNG = rng

        # --- 初期集団生成 ------------------------------------------
        if init_genes is not None:
            self.genes = init_genes.copy()
        else:
            if n_individuals is None or dim is None:
                raise ValueError("init_genes を与えない場合は n_individuals と dim が必要です")
            # bounds 解釈
            if bounds is None:
                lo, hi = 0.0, 1.0
                low = np.full(dim, lo)
                high = np.full(dim, hi)
            elif isinstance(bounds[0], (int, float)):
                lo, hi = bounds  # type: ignore[misc]
                low = np.full(dim, lo)
                high = np.full(dim, hi)
            else:
                low = np.array([b[0] for b in bounds])
                high = np.array([b[1] for b in bounds])
            self.genes = rng.uniform(low, high, size=(n_individuals, dim))

        # 適応度計算
        self.fitness_fn = fitness_fn
        self.fitness = np.apply_along_axis(fitness_fn, 1, self.genes)

        # 履歴
        self.gene_history: List[Tuple[int, Array]] = []

    # ------------------------------------------------------------------
    def best(self) -> Tuple[Array, float]:
        idx = int(np.argmin(self.fitness))
        return self.genes[idx], float(self.fitness[idx])

    # internal ----
    def _next_generation(self, *, selector: SelectOp, crossover_op: CrossOp, mutation_op: MutOp, crossover_rate: float) -> None:
        pop_n, _ = self.genes.shape
        new = np.empty_like(self.genes)
        new[0] = self.best()[0]
        i = 1
        while i < pop_n:
            p1 = self.genes[selector(self.fitness, self.rng)]
            p2 = self.genes[selector(self.fitness, self.rng)]
            c1, c2 = (crossover_op(p1, p2, self.rng) if self.rng.random() < crossover_rate else (p1.copy(), p2.copy()))
            new[i] = mutation_op(c1, self.rng)
            if i + 1 < pop_n:
                new[i + 1] = mutation_op(c2, self.rng)
            i += 2
        self.genes = new
        self.fitness = np.apply_along_axis(self.fitness_fn, 1, self.genes)

    # public ----
    def evolve(
        self,
        generations: int,
        *,
        selector: SelectOp = tournament_select,
        crossover_op: CrossOp | None = None,
        mutation_op: MutOp | None = None,
        crossover_rate: float = 0.9,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
        bounds: Bounds = None,
        patience: int = 100,
        tol: float = 1e-8,
        target_fit: float | None = None,
        record_every: int | None = None,
        enable_early_stop: bool = True,
        verbose: bool = False,
    ) -> Tuple[Array, float]:
        # 演算子デフォルト
        if crossover_op is None:
            crossover_op = _noop_crossover if self.genes.shape[1] < 2 else lambda a, b, r: one_point(a, b, rng=r)
        if mutation_op is None:
            mutation_op = lambda x, r: gaussian(x, sigma=mutation_sigma, prob=mutation_prob, bounds=bounds, rng=r)  # type: ignore[override]

        best_prev = self.best()[1]
        stagnate = 0
        if record_every and record_every > 0:
            self.gene_history.append((0, self.genes.copy()))

        for g in range(1, generations + 1):
            self._next_generation(selector=selector, crossover_op=crossover_op, mutation_op=mutation_op, crossover_rate=crossover_rate)
            if record_every and g % record_every == 0:
                self.gene_history.append((g, self.genes.copy()))

            best_now = self.best()[1]
            if verbose and (g % 10 == 0 or g == generations):
                print(f"[gen {g:4d}] best fitness = {best_now:.4g}")

            if enable_early_stop:
                if target_fit is not None and best_now <= target_fit:
                    break
                if abs(best_prev - best_now) < tol:
                    stagnate += 1
                    if stagnate >= patience:
                        break
                else:
                    stagnate = 0
                best_prev = best_now
        return self.best()
