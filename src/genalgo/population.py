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
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence, Tuple, List

import numpy as np
from .selection import tournament_select
from .crossover import one_point
from .mutation import gaussian

# ---------------------------------------------------------------------
# 型エイリアス
# ---------------------------------------------------------------------
Array = np.ndarray
Bounds = Tuple[float, float] | Sequence[Tuple[float, float]] | None
RNG = np.random.Generator

# ---------------------------------------------------------------------
# オペレータの Protocol（型インタフェース）
# ---------------------------------------------------------------------
class SelectOp(Protocol):
    def __call__(self, fitness: Array, rng: RNG) -> int: ...

class CrossOp(Protocol):
    def __call__(self, p1: Array, p2: Array, rng: RNG) -> Tuple[Array, Array]: ...

class MutOp(Protocol):
    def __call__(self, individual: Array, rng: RNG) -> Array: ...

# 1 遺伝子用のデフォルト交叉（何もしない）

def _noop_crossover(a: Array, b: Array, rng: RNG) -> Tuple[Array, Array]:
    return a.copy(), b.copy()

# ---------------------------------------------------------------------
# Population クラス
# ---------------------------------------------------------------------
class Population:
    """遺伝的アルゴリズム (GA) 用の軽量エンジン。

    Parameters
    ----------
    init_genes : ndarray, shape = (N, dim)
        初期集団。各行が 1 個体の遺伝子ベクトルです。
    fitness_fn : Callable[[ndarray], float]
        個体 → スカラー適応度を返す関数（小さいほど良いと仮定）。
    rng : numpy.random.Generator, optional
        乱数生成器。省略時は `np.random.default_rng()` が生成されます。

    Notes
    -----
    * 適応度評価は `np.apply_along_axis` でベクトル化されます。
    * 演算子はデフォルトで `tournament_select` / `one_point` / `gaussian`。
      パラメータ `evolve()` で自由に置き換え可能です。
    * `record_every` を指定すると、指定世代ごとに genes のコピーを
      `self.gene_history` に保存します。メモリ消費には注意してください。
    """

    # ------------------------------ 初期化 ----------------------------
    def __init__(
        self,
        init_genes: Array,
        fitness_fn: Callable[[Array], float],
        *,
        rng: RNG | None = None,
    ) -> None:
        self.rng: RNG = rng or np.random.default_rng()
        self.genes: Array = init_genes.copy()
        self.fitness_fn = fitness_fn
        self.fitness: Array = np.apply_along_axis(fitness_fn, 1, self.genes)

        # 履歴格納用リスト (gen, genes)
        self.gene_history: List[Tuple[int, Array]] = []

    # ------------------------- ユーティリティ ------------------------
    def best(self) -> Tuple[Array, float]:
        """最良個体とその適応度を返す。"""
        idx = int(np.argmin(self.fitness))
        return self.genes[idx], float(self.fitness[idx])

    # ---------------------- 1 世代更新処理 ---------------------------
    def _next_generation(
        self,
        *,
        selector: SelectOp,
        crossover_op: CrossOp,
        mutation_op: MutOp,
        crossover_rate: float,
    ) -> None:
        """内部用 — 1 世代分の更新を行う。"""
        pop_n, _ = self.genes.shape
        new_genes = np.empty_like(self.genes)

        # --- エリート保持（index 0 にコピー）
        new_genes[0] = self.best()[0]

        i = 1
        while i < pop_n:
            p1 = self.genes[selector(self.fitness, self.rng)]
            p2 = self.genes[selector(self.fitness, self.rng)]

            # 交叉
            if self.rng.random() < crossover_rate:
                c1, c2 = crossover_op(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # 変異
            new_genes[i] = mutation_op(c1, self.rng)
            if i + 1 < pop_n:
                new_genes[i + 1] = mutation_op(c2, self.rng)
            i += 2

        # 集団更新 & 適応度再計算
        self.genes = new_genes
        self.fitness = np.apply_along_axis(self.fitness_fn, 1, self.genes)

    # ------------------------- 進化メイン ----------------------------
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
        verbose: bool = False,
    ) -> Tuple[Array, float]:
        """GA を実行して最良個体を返す。

        Parameters
        ----------
        generations : int
            最大世代数。
        selector, crossover_op, mutation_op : callable, optional
            独自演算子を注入したい場合に指定。
        crossover_rate : float
            交叉を適用する確率 (0–1)。
        mutation_prob, mutation_sigma : float
            `gaussian` 変異用のパラメータ。独自 mutation を使うなら無視されます。
        bounds : tuple or None
            変異後にクリップする範囲。グローバル (lo, hi) または
            各遺伝子ごとのリスト `[(lo0, hi0), ...]`。
        patience, tol : int, float
            改善停滞による早期停止のしきい値。
        target_fit : float or None
            適応度がこの値以下になったら停止。
        record_every : int or None
            *n* を指定すると *n* 世代ごとに遺伝子行列を
            `self.gene_history` に保存します。メモリに注意。
        verbose : bool
            True で 10 世代ごとに進捗を表示。
        """
        # --- デフォルト演算子決定 -----------------------------------
        if crossover_op is None:
            crossover_op = (
                _noop_crossover
                if self.genes.shape[1] < 2
                else lambda a, b, r: one_point(a, b, rng=r)
            )

        if mutation_op is None:
            def mutation_op(x: Array, rng: RNG) -> Array:  # type: ignore[override]
                return gaussian(x, sigma=mutation_sigma, prob=mutation_prob, bounds=bounds, rng=rng)

        # --- 進化ループ ---------------------------------------------
        best_prev = self.best()[1]
        stagnate = 0

        if record_every == 0:
            record_every = None  # 0 は無効扱い
        if record_every is not None:
            # 世代 0 (初期) を保存
            self.gene_history.append((0, self.genes.copy()))

        for g in range(generations):
            self._next_generation(
                selector=selector,
                crossover_op=crossover_op,
                mutation_op=mutation_op,
                crossover_rate=crossover_rate,
            )

            # 履歴記録
            if record_every is not None and (g + 1) % record_every == 0:
                self.gene_history.append((g + 1, self.genes.copy()))

            # 進捗表示
            best_now = self.best()[1]
            if verbose and (g % 10 == 9 or g == generations - 1):
                print(f"[gen {g+1:4d}] best fitness = {best_now:.4g}")

            # 停止判定
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
