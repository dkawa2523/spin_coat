# Benchmark Evaluation (Numerical Only, 300-Trial Sweep)

データ: `data/benchmark/*.csv`  
実行: `venv/bin/python run_benchmark_suite.py --configs config_benchmark.yaml config_benchmark_euler_numpy.yaml config_benchmark_rk23_numpy.yaml config_benchmark_rk4_jax.yaml config_benchmark_constrained.yaml config_benchmark_grad.yaml --trials 300 --output bench_report.md`  
成果物: 各`outputs_benchmark*`ディレクトリ（設定別）と`bench_report.md`

## 1. NumPy + 積分器比較（Optuna, 300 trials）

| integrator | best MSE | duration [s] | completed | output dir |
| --- | --- | --- | --- | --- |
| euler | 4.971e-16 | 2.7 | 300 | outputs_benchmark_euler_numpy |
| rk23 (adaptive) | 6.771e-16 | 4.7 | 300 | outputs_benchmark_rk23_numpy |
| rk4 | 2.720e-15 | 4.5 | 300 | outputs_benchmark |

考察: 300試行ではEulerが最小MSEを達成し計算時間も最短。RK23は僅差で精度良好。RK4は安定だがこのデータでは若干劣後。

## 2. NumPy vs JAX（RK4, Optuna, 300 trials）

| backend | best MSE | duration [s] | completed | output dir |
| --- | --- | --- | --- | --- |
| numpy rk4 | 2.720e-15 | 4.5 | 300 | outputs_benchmark |
| jax rk4 (cpu) | 5.690e-16 | 33.6 | 300 | outputs_benchmark_rk4_jax |

考察: 精度はJAXがわずかに良いが、初期コンパイルと実行オーバーヘッドで時間は約7倍。大量バッチやGPUを使う場合にメリットが出る見込み。

## 3. 制約あり/なし（連続性正則化, weight=1e-2）

| continuity | best MSE | duration [s] | completed | output dir |
| --- | --- | --- | --- | --- |
| off (w=0) | 2.720e-15 | 4.5 | 300 | outputs_benchmark |
| on  (w=0.01) | 7.361e-16 | 7.4 | 300 | outputs_benchmark_constrained |

考察: 正則化ONでMSEが改善しつつ時間はやや増。滑らかな係数推定とスコアを両立できており、解釈性重視ならONが有利。

## 4. 最適化手法比較

| optimiser | backend / integrator | best MSE | duration [s] | trials/iters | output dir |
| --- | --- | --- | --- | --- | --- |
| optuna | numpy / euler | 4.971e-16 | 2.7 | 300 | outputs_benchmark_euler_numpy |
| optuna | numpy / rk23 | 6.771e-16 | 4.7 | 300 | outputs_benchmark_rk23_numpy |
| optuna | numpy / rk4 | 2.720e-15 | 4.5 | 300 | outputs_benchmark |
| optuna | jax / rk4 | 5.690e-16 | 33.6 | 300 | outputs_benchmark_rk4_jax |
| gradient | numpy / rk4 | 1.702e-14 | 0.2 | 1 | outputs_benchmark_grad |

考察: 大域探索（Optuna）が全体的に優秀。勾配法は即時に終わるが精度は桁落ち。計算時間を抑えつつ精度も高いのは「Optuna + Euler or RK23」。JAXは精度は良いが時間コストを考慮して選択。

## 5. 追加メモ

- 詳細ログと表は`bench_report.md`に出力済み（同上コマンドで生成）。
- ディレクトリは設定ごとに分離済み（`outputs_benchmark*`）ので、追加試行を重ねても結果を混在させず管理可能。
