# Spin Coating Fitting (NumPy/JAX) – Technical Guide

本ドキュメントは、スピンコート膜厚モデルの理論背景、数値解法、最適化手法、コード構成をまとめた技術リファレンスです。JAX/NumPy いずれでも同一APIで利用できるようリファクタ済みです。

## 1. 理論式と有用性

### ODE (拡張 Meyerhofer–EBP)
\[
  \frac{dh_i}{dt} = -\,C_i\,\frac{h_i^3}{\mu(h_i)} - E_i(h_i)
\]
- \(h_i(t)\): 計測点 \(i\) の膜厚  
- \(C_i = k_i \frac{2\rho\omega^2}{3}\): 流動項係数（回転数・密度・位置補正）  
- \(\mu(h) = \mu_0 (h/h_\mathrm{ref})^{-m}\): 膜厚依存粘度（薄膜ほど粘度上昇をモデル化）  
- \(E_i(h) = (1+\delta E_i) E_0 (h/h_\mathrm{ref})^{\alpha_E}\): 蒸発項（位置補正＋膜厚依存）  

**近似性と有用性**  
- ニュートン流体＋薄膜近似＋一様回転場を前提とした簡約モデル。  
- 蒸発は厚み依存のべき乗近似。温度・濃度・スキン形成などの一次効果を吸収しやすい。  
- 物性差・位置差を少数パラメータ (k_i, δE_i) で吸収でき、実験再現やスケーリング検討に有用。  

**参考文献**  
- Meyerhofer, D. “Characteristics of Resist Films Produced by Spinning.” *J. Appl. Phys.* 49, 3993–3997 (1978).  
- Bornside et al. “Spin coating of a colloidal suspension.” *Phys. Fluids* 1, 179–187 (1989).  
- Emslie, Bonner, Peck (EBP). “Flow of a viscous liquid…” *J. Appl. Phys.* 29, 858 (1958).  

## 2. 変数の分類（最適化対象 vs プロセス条件）

| 区分 | 変数 | 役割 | 変更タイミング |
| --- | --- | --- | --- |
| プロセス条件 | ρ (density), ω (rad/s), h_ref | 流動スケール・無次元化基準 | 実験条件が変わるとき |
| モデル係数（グローバル） | log10_mu0, m, log10_E0, alpha_E | 物性・蒸発の基準と厚み依存性 | 試料/溶媒/温度が変わるときに最適化 |
| 位置補正（センサー別） | k_flow_i, delta_E_i | 流動・蒸発の空間差を吸収 | 基板位置差が顕著なときに最適化 |
| 初期条件 | initial_h (from_data/fixed) | スピン開始時の膜厚 | 計測の取得方法に応じて設定 |

## 3. パラメータの物理的意味と効果（model.md に基づく）

| パラメータ | 物理的意味 | h(t)への主効果 | 変更・最適化が効くケース |
| --- | --- | --- | --- |
| log10_mu0 | 初期厚み付近の基準粘度 | 全体の流動速度スケール | 材料や温度が変わったとき |
| m | 膜厚依存の粘度指数 | 薄膜域での流動抑制/促進 | 後期の曲率が合わない |
| log10_E0 | 基準蒸発速度 | 後期の直線傾き | 環境/溶媒で乾燥速度が変化 |
| alpha_E | 蒸発の厚み依存 | 後期傾きの時間変化 | 後半で急乾/停滞が見られる |
| k_flow_i | 位置別流動補正 | 初期〜中期の速度差（位置依存） | edge/middle/center の速度差 |
| delta_E_i | 位置別蒸発補正 | 後期傾きの位置差 | 風向/温度むらで乾燥差が顕著 |

## 4. コード構成

| ファイル/モジュール | 役割 |
| --- | --- |
| `spinfit/types.py` | パラメータ型定義 (GlobalParams, SensorParams) |
| `spinfit/equations.py` | μ(h), E(h), rhs の実装（拡張/差替しやすい） |
| `spinfit/integrators.py` | ODE積分器 (euler, rk4, rk23, semi_implicit) を NumPy/JAX 両対応で登録 |
| `spinfit/solvers.py` | backend+integratorを束ねて rhs を解く共通 Solver |
| `spinfit/parameters.py` | pack/unpack/build_params_from_theta |
| `spinfit/losses.py` | MSE + 連続性正則化（NaN/Inf ペナルティ） |
| `spinfit/optimize.py` | Optuna 探索（fANOVA 重要度、サンプラ選択対応） |
| `spinfit/simulation.py` | 固定パラメータで各センサー h(t) を生成 |
| `run_optuna.py` | Optuna 実行 CLI |
| `run_grad.py` | 有限差分勾配降下 CLI |
| `run_forward.py` | 固定パラメータの前向き計算 CLI |
| `calcuration.py` | 統合ランナー (run_mode: optimize/forward) |
| `plot_utils.py` | Optuna診断プロット/HTML、重要度フィルタリング、全トライアルCSV |

## 5. 数値処理ワークフロー (mermaid)

```mermaid
flowchart TD
  A[Load YAML] --> B[Select mode (run_mode/forward.enabled/optimiser)]
  B -->|forward| F[Build fixed params & solver] --> G[Simulate h(t)]
  B -->|optuna/grad| C[Load data & time grids]
  C --> D[Suggest/Update theta]
  D --> E[Solver predicts h(t)]
  E --> H[Loss calc (MSE + continuity)]
  H -->|Optuna| D
  H -->|Grad descent| D
  G --> I[Summaries/plots]
  H --> J[Diagnostics: history/parallel/importance/trials CSV]
```

### 各ステップの役割・効果
- Load YAML: backend/integrator/initial_h/continuity など全設定を集約
- Mode selection: forward と最適化を明確に分離
- Suggest/Update theta: Optuna がグローバル探索、GD が局所微調整
- Solver predicts: NumPy/JAX + 選択積分器で h(t) 計算
- Loss calc: MSE と連続性正則化で発散を抑止
- Diagnostics: 重要度/履歴/試行表でパラメータ感度と収束性を可視化

## 6. 数値計算法の比較

| 手法 | メリット | デメリット | 使い所 |
| --- | --- | --- | --- |
| Euler | 実装簡単・高速 | 精度/安定性が低め | 粗い探索や初期テスト |
| RK4 | 高精度・安定 | ステップ固定でコスト増 | 精度重視、ステップ十分小さい場合 |
| RK23 (Adaptive) | 自動ステップ調整、効率的 | JIT非対応（JAXではPythonループ） | 精度と効率の折衷、硬くない系 |
| Semi-implicit | 簡易安定化、硬い項に強い | 収束保証なし、固定反復 | 蒸発項が強く硬いときの安全策 |

## 7. NumPy vs JAX

| backend | メリット | デメリット | 使い所 |
| --- | --- | --- | --- |
| NumPy | 安定・依存少・デバッグ容易 | 大規模並列なし | CPUメイン、小規模/迅速検証 |
| JAX | JITで高速、GPU/TPU利用可 | 初期オーバーヘッド、依存重い | 大規模試行、高速推論、ML拡張 |

## 8. 最適化手法

| 手法 | メリット | デメリット | 使い所 |
| --- | --- | --- | --- |
| Optuna (TPE/CMAES) | 大域探索に強い、分布更新 | 試行コスト大 | 広い範囲の同定、初期探索 |
| 勾配降下 (有限差分) | 実装簡単、局所微調整 | ノイズに弱い、ステップ調整要 | 既知近傍での微調整 |

## 9. 実行方法（用途別）

| 用途 | 設定 | コマンド |
| --- | --- | --- |
| Optuna最適化 | `run_mode: optimize`, `fit.optimiser: optuna` | `./.venv/bin/python calcuration.py --config config_example.yaml --trials 50 --timeout 300` |
| 勾配法 | `run_mode: optimize`, `fit.optimiser: gradient` | `./.venv/bin/python calcuration.py --config config_example.yaml` |
| 前向き計算 | `run_mode: forward` または `forward.enabled: true` | `./.venv/bin/python calcuration.py --config config_example.yaml` |

## 10. YAML設定の要点

| キー | 意味 / 内容 | 変えるべき状況 |
| --- | --- | --- |
| run_mode | optimize / forward | 用途切替 |
| fit.backend | numpy / jax | 実行環境/GPU有無で選択 |
| fit.integrator | euler / rk4 / rk23 / semi_implicit | 精度・安定性・速度のバランス調整 |
| fit.initial_h | from_data / fixed | 初期膜厚取得方法 |
| fit.parameters.* | 最適化対象の範囲・初期値 | 材料/環境が変わるとき |
| fit.optuna.* | n_trials, sampler, timeout 等 | 探索深さ・時間制限を調整 |
| fit.gradient.* | lr, max_iters, grad_eps 等 | 局所微調整の収束性改善 |
| fit.continuity.* | 連続性正則化の有無/重み | 振動解・発散を抑えたい |
| forward.enabled | true で前向き計算実行 | 最適化不要で条件計算したい |
| forward.parameters | 固定パラメータ指定 | 同定済み値を直接使う |
| forward.override_model | rho/omega/h_ref 上書き | プロセス条件を変えて計算 |

---

以上の構造により、方程式・係数・損失・積分器・可視化・実行モードを独立に拡張でき、JAX/NumPyの切り替えやML統合も容易です。***
