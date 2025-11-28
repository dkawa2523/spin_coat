# Spin Coating Film Thickness Fitting Project

This project provides a simple yet flexible framework for fitting time‐dependent spin coating film thickness data using a set of ordinary differential equations (ODEs).  The primary goal is to make it easy to experiment with different physical models, compare backends (NumPy vs JAX), and tune parameters either with gradient descent or with [Optuna](https://optuna.org/) for hyperparameter optimization.

## Contents

* `model.py` – Core definitions for the ODE model, parameter packing/unpacking, loss functions, and solvers.  This file also exposes classes for both NumPy and JAX backends and includes a simple gradient descent optimiser.
* `run_optuna.py` – Entry point for running an Optuna search using a YAML configuration file.  It loads data, builds models, and prints the best parameters.
* `run_grad.py` – Entry point for running the built‑in gradient descent optimiser.  It shares a nearly identical API to `run_optuna.py` but uses finite differences rather than Optuna.
* `config_example.yaml` – Example configuration file showing how to specify data locations, initial guesses, search ranges, and optimiser settings.
* `data/` – Directory containing synthetic test data for three sensor locations: `center`, `middle`, and `edge`.  These CSV files have two columns (no header): `time` and `thickness` (in metres).

## Usage

1. **Install dependencies**

   Within a virtual environment, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data**

   Place your experimental time series for each sensor into the `data/` directory.  Each file should be a CSV without a header, containing two columns: time (s) and film thickness (m).  Update the `config_example.yaml` file accordingly to reference your files.

3. **Run Optuna optimisation**

   ```bash
   python run_optuna.py --config config_example.yaml
   ```

   This will perform a hyperparameter search and return the best set of parameters.  You can adjust the search ranges and the number of trials in the YAML file.

4. **Run gradient descent optimisation**

   ```bash
   python run_grad.py --config config_example.yaml
   ```

   This runs a simple finite difference gradient descent.  You can adjust learning rate, maximum iterations, and stopping criteria in the YAML file.

Both entrypoints resolve `data.base_dir` relative to the location of the YAML config.  `run_optuna.py` also accepts `--trials` and `--timeout` flags to override the config for quick experiments.

5. **Plotting best-fit curves (Optuna)**

   By default `run_optuna.py` saves data vs. model plots to `outputs/optuna_fit_<sensor>.png`.  Disable via `--no-plot` if you only need text output.

## Batch evaluation & 異域（転移的）学習の検証方法

- **数値計算のみの一括ベンチマーク**（任意データでも同じ手順）  
  1) `config_benchmark*.yaml` を対象データに合わせて `data.base_dir` とファイル名を更新  
  2) 実行:  
     ```bash
     venv/bin/python run_benchmark_suite.py \
       --configs config_benchmark.yaml config_benchmark_euler_numpy.yaml config_benchmark_rk23_numpy.yaml \
       config_benchmark_rk4_jax.yaml config_benchmark_constrained.yaml config_benchmark_grad.yaml \
       --trials 300 --output bench_report.md
     ```  
  3) まとめ結果: `bench_report.md`（表）、詳細成果物: 各 `outputs_benchmark*` ディレクトリ。`test.md` には視点別（積分器・NumPy/JAX・制約有無・最適化手法）の考察付きサマリーを保持。

- **異域/転移的な学習（Hybrid Neural ODE, JAX）**  
  - 設定: `training_config.yaml` で学習専用設定を記述（学習用データ、ステップ数、学習率、目標R2など）。ベース設定は `config_example.yaml` を使い、学習時のみ上書きする。  
  - 学習実行:  
    ```bash
    venv/bin/python run_neural_ode.py \
      --config config_example.yaml \
      --train-config training_config.yaml \
      --enable
    ```  
    学習曲線、R2/MSE可視化、パラメータ (`params.npz`) は `neural_ode.output_dir` に保存。  
  - 別ドメイン/ベンチマークデータで再検証する場合は、`training_config.yaml` の `data.base_dir` とファイル名を差し替え、同コマンドを再実行する。  
  - 推論のみ行う場合は、`neural_ode.inference.enabled: true` と `params_path` を学習済み `params.npz` に設定し、`run_neural_ode.py --config <base> --enable` を実行（再学習せず評価だけ実施）。

## Configuration notes

- `fit.optimiser`: choose `optuna` or `gradient` to select which runner should execute. The non-selected runner will exit with a message.
- `fit.output_dir`: directory (relative to the config) where plots and reports are written.
- Optuna diagnostics now include PNG + HTML (Plotly) outputs: optimisation history (loss on log scale), parallel coordinates, parameter importances (overall + per-sensor HTML), trial durations, and a CSV of the top 10 trials.
- `fit.integrator`: choose ODE integrator (NumPy/JAX共通 API)
  - `euler`: simple, fast
  - `rk4`: 4th-order Runge–Kutta
  - `rk23`: 自動ステップ調整付きRK23（Bogacki–Shampine）
  - `semi_implicit`: 簡易な半陰的ステップ（固定反復）
  - 重要度計算は極端なペナルティ試行や上位外れ値を除外したうえで (>=8 安定試行) fANOVA を実施し、偏りを緩和
- `forward`: optimisationと切り分けた固定パラメータの前向き計算モード。`run_forward.py`で使用。
  - `forward.enabled`: trueで前向き計算を実行
  - `forward.output_dir`: 結果出力ディレクトリ
  - `forward.integrator`: euler/rk4/rk23/semi_implicit
  - `forward.override_model`: rho/omega/h_refの上書き（プロセス条件変更用）
  - `forward.parameters`: log10_mu0/m/log10_E0/alpha_E + センサー別 k_flow, delta_E を固定指定
  - `forward.initial_h`: from_data or fixed
- `run_mode`: `optimize` または `forward` を指定すると、`calcuration.py` で明示的にモードを選択できる（forward.enabled も有効であれば前向き計算優先）

## Model/Implementation highlights (from model.md and code)

| Model要素 | コード上の工夫 | 有用性 |
| --- | --- | --- |
| ODE構造: \\(dh/dt = -C_i h^3 / \mu(h) - E_i(h)\\) | `rhs_ode`で流動/蒸発項を明示し、`C_i=k_i*(2ρΩ²/3)`と蒸発補正`delta_E_i`をセンサー別に保持 | 流動・蒸発を分離したままパラメータ操作でき、model.mdの物理解釈を保ったフィットが可能 |
| 膜厚依存粘度 \\(\mu=\mu_0 (h/h_ref)^{-m}\\) | `mu_model`でクリップ付き計算、`log10_mu0`パラメータで数値安定化 | 薄膜側での発散を防ぎつつオーダーを扱いやすくし、探索の収束を安定化 |
| 蒸発モデル \\(E_i=(1+\delta_E_i)E_0 (h/h_ref)^{\alpha_E}\\) | `evap_model`で厚み依存と位置補正を切り分け、`alpha_E`省略時は自動で0に | 流動支配フェーズ/蒸発支配フェーズを段階的に調整しやすく、位置差の後期挙動を最小自由度で表現 |
| 数値積分 | NumPy: 前進Euler、JAX: `lax.scan`によるJITフレンドリな積分器 | JAXでもPythonループを避け高速化し、backend切替で同一APIを維持 |
| 時間グリッド/初期条件 | データ時刻をそのまま使うか一様グリッドを選択可、初期膜厚はデータ先頭 or 固定値 | model.mdで推奨される初期条件固定と領域別重み付けを実践しやすい |
| 最適化・探索 | Optuna: configで範囲・試行数を定義、CMA-ES/TPE切替可。Gradient: 有限差分GDを共通APIで利用 | 大域探索と局所微調整の両方をサポートし、段階的フィット戦略をコードで再現 |
| 安定性 | JAX向けに`JAX_ENABLE_X64`設定、GPU指定が無い場合はCPUに自動フォールバック。NaN/Inf時は損失にペナルティ | NumPy/JAX間の損失スケール差を縮小し、発散トライアルを早期に無効化 |
| 出力と可視化 | 予測 vs データPNG、Optunaの履歴・並行座標・重要度・試行時間のPNG/HTML、上位10トライアルCSV | フィットの収束挙動とパラメータ感度を可視化し、次のモデル拡張の判断材料を提供 |

## Extending the model

The code is designed to be easy to extend.  The core physical model is defined in a few functions:

* `mu_model(h, gp, xp)` – returns the viscosity as a function of film thickness.  It currently implements a power‑law dependence but can be replaced or extended with your own model.
* `evap_model(h, sp, gp, xp)` – returns the evaporation rate.  Currently it supports a base rate with an optional power dependence on thickness; you can add temperature dependence or other effects here.
* `rhs_ode(h, t, sp, gp, xp)` – defines the ODE itself.  Modifying this function lets you add or remove terms, include spatial coupling, etc.

When adding new parameters, make sure to update the parameter packing/unpacking functions and the YAML configuration accordingly.  All parameters live in a single vector (`theta`) for easy optimisation.

## Synthetic data

The repository includes synthetic datasets for demonstration purposes.  They were generated by numerically integrating the same ODEs using known “true” parameters and adding a small amount of Gaussian noise.  The parameters used are documented in the `run_optuna.py` and `run_grad.py` scripts.  You can examine these scripts to understand how the synthetic data were produced.

Feel free to modify or replace the synthetic data with your own measurements.
