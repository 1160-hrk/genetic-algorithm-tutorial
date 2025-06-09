"""src/genealgo/population.py  —  汎用 GA エンジン（履歴記録機能付き）
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
# ==============================================================
# 2025/0608 17:00 再改修
# 汎用 GA エンジン  +  自己適応変異サポート
#
# * self_adaptive=True にすると
#       ・self.sigmas (N,) を内部保持
#       ・mutation_op が無視され、gaussian_sa が呼ばれる
# * 既存コードは self_adaptive=False（既定）でそのまま動作
# ==============================================================
================================================================

* self_adaptive=True で個体ごとに σ を進化 (`gaussian_sa`)
* record_every=N で N 世代ごとに genes を履歴保存
* tau_scale / sigma_floor で自己適応パラメータを外から調整
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Protocol, Sequence, Tuple, List

from .selection import tournament_select
from .crossover import one_point
from .mutation import gaussian, gaussian_sa

Array  = np.ndarray
Bounds = Tuple[float, float] | Sequence[Tuple[float, float]] | None
RNG    = np.random.Generator


# ---------- operator protocols ----------
class SelectOp(Protocol):
    def __call__(self, fitness: Array, rng: RNG) -> int: ...

class CrossOp(Protocol):
    def __call__(self, p1: Array, p2: Array, rng: RNG) -> Tuple[Array, Array]: ...

class MutOp(Protocol):
    def __call__(self, ind: Array, rng: RNG) -> Array: ...


def _noop_crossover(a: Array, b: Array, rng: RNG) -> Tuple[Array, Array]:
    return a.copy(), b.copy()


# ---------------- Population ------------
class Population:
    """GA 集団クラス（固定σ or 自己適応σ をフラグで切替）"""

    # --- static default selector ----------------------------------
    @staticmethod
    def _default_selector(fitness: Array, rng: RNG) -> int:
        return tournament_select(fitness, k=3, rng=rng)

    # --- ctor -----------------------------------------------------
    def __init__(
        self,
        *,
        fitness_fn: Callable[[Array], float],
        # 初期集団（行列）または自動生成パラメータ
        init_genes: Array | None = None,
        n_individuals: int | None = None,
        dim: int | None = None,
        bounds: Bounds = (0.0, 1.0),
        # RNG
        seed: int | None = None,
        rng: RNG | None = None,
        # 自己適応 GA オプション
        self_adaptive: bool = False,
        init_sigma: float = 0.1,
        sigma_floor: float = 1e-5,
        tau_scale: float = 0.3,
    ):
        self.rng: RNG = rng or np.random.default_rng(seed)

        # ----- genes ----------
        if init_genes is not None:
            self.genes = init_genes.copy()
        else:
            if n_individuals is None or dim is None:
                raise ValueError("init_genes を与えない場合は n_individuals と dim が必要")
            if isinstance(bounds[0], (int, float)):
                lo, hi = bounds  # type: ignore[misc]
                low  = np.full(dim, lo)
                high = np.full(dim, hi)
            else:
                low  = np.array([b[0] for b in bounds])
                high = np.array([b[1] for b in bounds])
            self.genes = self.rng.uniform(low, high, size=(n_individuals, dim))

        # ----- self-adaptive σ ----------
        self.self_adaptive = self_adaptive
        self.sigmas = np.full(len(self.genes), init_sigma) if self_adaptive else None  # type: ignore
        self.sigma_floor = sigma_floor
        self.tau_scale   = tau_scale
        self.bounds      = bounds

        # ----- fitness ----------
        self.fitness_fn = fitness_fn
        self.fitness = np.apply_along_axis(fitness_fn, 1, self.genes)

        # ----- history ----------
        self.gene_history: List[Tuple[int, Array]] = []

    # -------- utilities -----------------
    def best(self) -> Tuple[Array, float]:
        idx = int(np.argmin(self.fitness))
        return self.genes[idx], float(self.fitness[idx])

    # -------- one generation ------------
    def _next_generation(
        self,
        *,
        selector: SelectOp,
        crossover_op: CrossOp,
        mutation_op: MutOp,
        crossover_rate: float,
    ) -> None:
        N, _ = self.genes.shape
        new_genes = np.empty_like(self.genes)
        new_genes[0] = self.best()[0]  # elite

        i = 1
        while i < N:
            p1 = self.genes[selector(self.fitness, self.rng)]
            p2 = self.genes[selector(self.fitness, self.rng)]
            c1, c2 = (crossover_op(p1, p2, self.rng)
                      if self.rng.random() < crossover_rate else (p1.copy(), p2.copy()))

            if self.self_adaptive:
                self.genes[i] = c1
                if i + 1 < N:
                    self.genes[i + 1] = c2
                gaussian_sa(i, self, tau_scale=self.tau_scale, rng=self.rng)
                if i + 1 < N:
                    gaussian_sa(i + 1, self, tau_scale=self.tau_scale, rng=self.rng)
            else:
                new_genes[i] = mutation_op(c1, self.rng)
                if i + 1 < N:
                    new_genes[i + 1] = mutation_op(c2, self.rng)
            i += 2

        if not self.self_adaptive:
            self.genes = new_genes

        self.fitness = np.apply_along_axis(self.fitness_fn, 1, self.genes)

    # ------------- evolve --------------
    def evolve(
        self,
        generations: int,
        *,
        selector: SelectOp = _default_selector.__func__,  # static method unwrap
        crossover_op: CrossOp | None = None,
        mutation_op: MutOp | None = None,
        crossover_rate: float = 0.6,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 0.1,
        patience: int = 100,
        tol: float = 1e-8,
        target_fit: float | None = None,
        record_every: int | None = None,
        enable_early_stop: bool = True,
        verbose: bool = False,
    ) -> Tuple[Array, float]:

        if crossover_op is None:
            crossover_op = (_noop_crossover if self.genes.shape[1] < 2
                            else lambda a, b, r: one_point(a, b, rng=r))

        if mutation_op is None:
            mutation_op = lambda x, r: gaussian(
                x, sigma=mutation_sigma, prob=mutation_prob,
                bounds=self.bounds, rng=r)  # type: ignore[override]

        best_prev = self.best()[1]
        stagnate  = 0
        if record_every:
            self.gene_history.append((0, self.genes.copy()))

        for g in range(1, generations + 1):
            self._next_generation(
                selector=selector,
                crossover_op=crossover_op,
                mutation_op=mutation_op,
                crossover_rate=crossover_rate,
            )

            # --- σ floor ---------------
            if self.self_adaptive and self.sigma_floor > 0.0:
                np.maximum(self.sigmas, self.sigma_floor, out=self.sigmas)

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
