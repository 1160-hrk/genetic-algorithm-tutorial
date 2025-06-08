# Crossover 演算子まとめ

> 本ページでは `src/genalgo/crossover.py` に実装されている **4 種類の交叉関数** を，数式・図・コード断片を交えて解説します。
> すべて「実数遺伝子（Real-coded GA）」向けの実装です。

---

## 目次

1. [one\_point](#one_point)
2. [uniform](#uniform)
3. [blx\_alpha](#blx_alpha)
4. [sbx](#sbx)
5. [比較と使い分け](#comparison)

---

<a name="one_point"></a>

## 1. `one_point`

```python
child1, child2 = one_point(parent1, parent2, rng)
```

| 特徴     | 内容                                             |
| ------ | ---------------------------------------------- |
| 切断位置   | 長さ `d` の場合 `1 ≤ cut ≤ d-1` を乱択                 |
| 子の生成   | `[p1[:cut], p2[cut:]]`, `[p2[:cut], p1[cut:]]` |
| 1 次元特例 | `d < 2` なら親をそのままコピー                            |

> **使い所**  : 高次元 (d ≥ 5) で遺伝子に「並び順の意味」がある場合（例: ポリゴン頂点の座標列）。

---

<a name="uniform"></a>

## 2. `uniform`

```python
child1, child2 = uniform(parent1, parent2, p=0.5, rng)
```

| パラメータ | 意味                        |
| ----- | ------------------------- |
| `p`   | 遺伝子ごとにスワップする確率（デフォルト 0.5） |

各遺伝子を独立にサイコロ判定して交換します。
**遺伝子間の順序依存が弱い** 問題（例: 実数ベクトルの重み最適化）に向きます。

---

<a name="blx_alpha"></a>

## 3. `blx_alpha`  (Blend Crossover, BLX-α)

```python
child1, child2 = blx_alpha(p1, p2, alpha=0.5, rng)
```

<!-- ![blx](docs/img/blx_alpha.svg) -->

* 各遺伝子ごとに区間 `[min − α⋅d,  max + α⋅d]` から一様乱数で子を生成。
  ここで `d = |p1 - p2|`。
* `α` を大きくすると **親の外側** の探索範囲が広がります。

> 初期探索フェーズで有効。局所詰めには α を下げるか、別演算子へ切替。

---

<a name="sbx"></a>

## 4. `sbx`  (Simulated Binary Crossover)

```python
child1, child2 = sbx(p1, p2, eta=2.0, rng)
```

式（Deb & Agrawal, 1995）：
```math
\beta = \begin{cases}
(2u)^{1/(\eta+1)} & u≤0.5\\
(1/(2(1-u)))^{1/(\eta+1)} & u>0.5
\end{cases}
```

子は
```math
c_1 = 0.5[(1+\beta)p_1 + (1-\beta)p_2]\quad
c_2 = 0.5[(1-\beta)p_1 + (1+\beta)p_2]
```

| `eta` 値 | 挙動                     |
|-----------|-------------------------|
| `η → 0`   | 子が親区間の外へ大きく飛ぶ |
| `η = 2`   | 論文推奨の標準値        |
| `η → ∞`   | 子 ≈ 親（小さく揺らぐだけ） |

> **連続最適化でデファクト。** `eta` を段階的に大きくすると擬似アニーリングになります。

---

<a name="comparison"></a>
## 5. どれを使えばいい？

| 問題タイプ            | 推奨演算子 | 理由                                    |
|-----------------------|------------|-----------------------------------------|
| 次元が低い (d ≤ 2)    | `sbx` or `blx_alpha(α=0.5)` | 親区間を外れても探索範囲が狭いので安全 |
| 順序依存が強い         | `one_point` | 遺伝子の並びが保たれやすい              |
| 序盤の大域探索         | `blx_alpha(α>0.5)` | 外側探索で局所最適を回避              |
| 終盤の微調整           | `sbx(eta≥5)` or `uniform(p<0.3)` | 親近傍で細かく詰める |

`Population.evolve()` の `crossover_op` にラムダで注入するだけで切り替えられます。

```python
pop.evolve(...,
           crossover_op=lambda a,b,r: sbx(a,b,eta=1.0,rng=r))
```

---

### 参考文献
* Deb, K., & Agrawal, R.B. (1995). **Simulated Binary Crossover for Continuous Search Space.** *Complex Systems,* 9(2), 115–148.
* Eshelman, L.J., & Schaffer, J.D. (1993). **Real-Coded Genetic Algorithms and Interval Schemata.** *Foundations of Genetic Algorithms 2*, 187–202.  （→ BLX‑α）

