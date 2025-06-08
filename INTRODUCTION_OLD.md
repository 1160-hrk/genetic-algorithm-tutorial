## 遺伝的アルゴリズム（GA）の全体像と実装の読み解きガイド

*(対象 : 大学院生／学部生)*

---

### 1. そもそも GA とは

1. **遺伝子表現** : 実数ベクトル（今回 dim = 1）。
2. **集団 (population)** : 遺伝子を複数まとめた行列 shape = (N, dim)。
3. **世代ループ**

   1. **評価 (fitness)** : 目的関数を各個体に作用させスカラー値を返す。
   2. **選択 (selection)** : いい個体ほど子を残しやすいよう親を選ぶ。
   3. **交叉 (crossover)** : 親 2 個体から子 2 個体を作る。
   4. **変異 (mutation)** : 各子の遺伝子にランダム微小ノイズを加える。
   5. **置換 (replacement)** : 新しい集団に刷新。エリート保持で最良個体は必ず残す。
4. **停止条件** : 世代上限・目標適応度到達・改善停滞など。

---

### 2. モジュール構成

```
genalgo/
├── selection.py   # 親選択演算子
├── crossover.py   # 交叉演算子
├── mutation.py    # 変異演算子
└── population.py  # GA ループを束ねるエンジン
rootfinder/
└── fitness.py     # |f(x)| ≒ 根からの誤差
```

#### 2.1 selection.py

```python
def tournament_select(fitness, k, rng): ...
```

* 集団から *k* 個を無作為抽出し、その中で最小 fitness (= 最良) を返す。
* 乱数シードを渡せるので再現性テストが容易。

#### 2.2 crossover.py

```python
def one_point(a, b, rng): ...
```

* 長さ≥2 のとき中央付近で切り替え、子を生成。
* 長さ 1 の場合は **noop**（親コピー）。

#### 2.3 mutation.py

```python
def gaussian(x, sigma, prob, bounds, rng): ...
```

* 各遺伝子に確率 *prob* で正規ノイズ 𝒩(0, σ²) を加算。
* 範囲外は `np.clip` で戻す。

---

### 3. population.py の詳細

| セクション                    | 役割                          | ポイント                                     |
| ------------------------ | --------------------------- | ---------------------------------------- |
| **型 alias**              | `Array`, `CrossOp`, `MutOp` | mypy で関数型チェックを通しやすくする                    |
| **コンストラクタ**              | 初期遺伝子をコピー → fitness 全計算     | `np.apply_along_axis` は (N, dim)→(N,) 一括 |
| **best()**               | 最良個体と fitness を返す           | `np.argmin` で一発取得                        |
| **\_next\_generation()** | 1 世代更新                      | ①親選択→②交叉→③変異→④評価                         |
| **evolve()**             | 高レベル API                    | 早期終了・進捗表示・演算子注入                          |

#### 3.1 Strategy パターン

```python
def evolve(..., crossover_op=None, mutation_op=None)
```

* 演算子関数を外から注入可能。
* 1 遺伝子なら `_noop_crossover` を自動選択。

#### 3.2 早期停止ロジック

```python
if abs(best_prev - best_now) < tol: stagnate += 1
if stagnate >= patience: break
```

* *tol* 未満の改善が *patience* 世代続く → 打ち切り。
* 目標適応度 `target_fit` もサポート。

---

### 4. 例 : cos x – x = 0 の根探し (`demo_root.py` 抜粋)

```python
f = lambda x: math.cos(x) - x
fitness = make_abs_fitness(f)          # |f(x)| を最小化
init = rng.uniform(0, 2, (100, 1))     # 初期集団
pop = Population(init, fitness, rng)
best_gene, best_fit = pop.evolve(
    generations=300,
    mutation_sigma=0.05,
    bounds=(0, 2),
    verbose=True,
)
```

* **結果** : ≈ 0.7391（真値 0.739085…）
* 出力ログの “No significant improvement” は早期停止が働いた証拠。

---

### 5. 何がロバストなのか？

| 改善点          | 失敗をどう避けるか                                         |
| ------------ | ------------------------------------------------- |
| **演算子の動的切替** | 次元が 1 → 自動で交叉を noop に。多次元・バイナリにも差し替え可。            |
| **型安全**      | `CrossOp/MutOp` を `Protocol` で定義 → mypy が署名ずれを検出。 |
| **エリート保持**   | 最良個体が突然死しない。                                      |
| **早期停止**     | 無益な計算を打ち切り、実行時間を節約。                               |
| **再現性**      | すべての RNG を `numpy.random.Generator` 経由で管理。        |

---

### 6. 物理系・数値最適化への応用例

| 応用                | 遺伝子の意味             | 変異の工夫                    |
| ----------------- | ------------------ | ------------------------ |
| **多峰性ポテンシャルの最小化** | 粒子座標 (x, y, z)     | アニーリング的に `sigma` を徐々に減らす |
| **分子の力場パラメータ同定**  | Lennard–Jones ε, σ | `bounds` で物理的に妥当な範囲に制限   |
| **レーザーパルス整形**     | 周波数成分の位相配列         | 位相は \[0, 2π) でラップして clip |

---

### 7. さらに拡張するなら

1. **選択演算子** : ルーレット選択・ランキング選択を追加。
2. **交叉** : BLX-α, SBX (Simulated Binary Crossover) など実数 GA 定番。
3. **多目的最適化** : NSGA-II のように Pareto front を直接求める。
4. **並列化** : 評価関数が重い場合 `numpy.apply_along_axis` を `joblib`・`dask` に置換。

---

## まとめ

今回の実装は **「小さく書き始めて安全に育てられる」** ことを意識しています。
*演算子を戦略として注入*、*型で拘束*、*早期停止で無駄打ち防止* ― これらは数値計算でありがちな **“パラメータ爆発 & 処理時間沼”** を避ける王道パターンです。
自分の研究課題に合わせて演算子を差し替えつつ、`tests/` を増やしていけば、**再現性ある数値実験プラットフォーム** として長く使えるはずです。
