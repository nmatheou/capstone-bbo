# Model Card: Human-in-the-Loop Hybrid Surrogate Optimizer (HitL-HSO)

## Overview
*   **Name of approach:** Human-in-the-Loop Hybrid Surrogate Optimizer (HitL-HSO)
*   **Type:** Sequential Black-Box Optimization (BBO) ensemble, combining Gaussian Process Regression (GPR), Shallow Neural Networks (PyTorch `64x64` ReLU MLP), Response Surface Methodology (RSM), and Human-in-the-Loop Diagnostics.
*   **Version:** 1.3 (Final Week 13 Capstone Submission)

---

## Model Description

**Input:**
*   **Historical Observation Matrices ($\mathbf{X}_t \in [0, 1]^{n \times d}, \mathbf{y}_t \in \mathbb{R}^n$):** Accumulated coordinate-evaluation pairs across $n \in [23, 53]$ samples for each of the 8 functions ($d \in \{2, 2, 3, 4, 4, 5, 6, 8\}$), normalised to zero mean and unit variance internally prior to fitting.
*   **Hyperparameter & Trust-Region Configuration (`configs/hyperparameters.json`):** Per-function acquisition parameters ($\kappa$ for UCB, $\xi$ for Expected Improvement), kernel smoothness ($\nu \in \{1.5, 2.5\}$), ARD length-scale bounds (`length_scale_bounds`), and adaptive trust-region bounds around the incumbent best.

**Output:**
*   **Predictive Posterior Distribution ($\mu(\mathbf{x}), \sigma(\mathbf{x})$):** Continuous mean and epistemic uncertainty predictions across the candidate space $[0, 1]^d$ from the Gaussian Process surrogate, paired with point predictions $\hat{y}_{\text{NN}}(\mathbf{x})$ from the PyTorch Neural Network surrogate.
*   **Next Sequential Query Coordinate ($\mathbf{x}_{t+1}^\ast \in [0, 1]^d$):** A single 6-decimal-formatted coordinate vector (`x1-x2-...-xd`) per function maximising the acquisition/refinement objective, written to `submissions/week_<N>/suggestions.txt`.

**Model Architecture:**
The architecture implemented in [`src/bbo_optimize.py`](../src/bbo_optimize.py) is a three-tier hybrid surrogate pipeline:
1.  **Tier 1 — Anisotropic Gaussian Process Regressor (`scikit-learn`):**
    *   Composite kernel $k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \, k_{\text{Matern}}(\mathbf{x}, \mathbf{x}'; \nu, \boldsymbol{\ell}) + \sigma_n^2 \delta(\mathbf{x}, \mathbf{x}')$ with **Automatic Relevance Determination (ARD)** length scales $\boldsymbol{\ell} = (\ell_1, \dots, \ell_d)$ optimised via multi-start L-BFGS-B marginal log-likelihood maximisation (`n_restarts_optimizer=10–25`).
    *   $\nu = 1.5$ (once-differentiable) for rugged landscapes ($F_4$); $\nu = 2.5$ (twice-differentiable) for smooth landscapes ($F_1, F_2, F_3, F_5, F_6, F_7, F_8$); explicit `WhiteKernel` noise regularisation for stochastic functions ($F_2, F_6$); and an RBF `SVC` active-subspace filter on $F_1$.
2.  **Tier 2 — Secondary Shallow Neural Network Surrogate (`PyTorch`):**
    *   A deterministic-seeded (`seed=42`) 2-hidden-layer Multi-Layer Perceptron (`d -> 64 -> ReLU -> 64 -> ReLU -> 1`) trained on z-score standardised targets using Adam (`lr=0.005`, `weight_decay=1e-4`, `800` epochs), followed by bounded L-BFGS input gradient ascent (`lr=0.05`, `20` steps) starting from the empirical incumbent best (`train_and_optimize_nn`).
3.  **Tier 3 — Analytical Local Response Surfaces & Diagnostic Refinement (`compute_final_refinement`):**
    *   Applies local positive-basin pruning ($y > 0$ on $F_4$; $[0.62, 0.67]^2$ on $F_1$), 2D log-Gaussian secant calibration ($F_1$), 2nd-order Quadratic Response Surface Methodology ($F_8$), and inactive-dimension anchoring ($d_2 = 0.387$ on $F_2$; $(d_1, d_2) = (0.95, 0.40)$ on $F_3$).

---

## Intended Use
*   **What tasks is it suitable for?** 
    This model is designed for high-cost, zero-derivative, severely budget-constrained black-box optimization tasks (e.g., physical sensor placement, experimental drug discovery, industrial recipe formulation, and expensive ML hyperparameter tuning) where data sparsity is extreme ($23 \le n \le 53$ points across 13 rounds, 275 total evaluations) and dimensions range from 2D to 8D.
*   **What use cases should be avoided?** 
    It is unsuitable for generating globally calibrated topological maps across the entire domain, as the late-stage acquisition strategy is hyper-focused on local summit exploitation. Furthermore, it should be avoided in fully autonomous, non-stationary continuous deployment systems without human oversight, as small-sample surrogates require diagnostic intervention to prevent boundary and covariance failure modes.

---

## Details (Strategy Evolution Across Thirteen Rounds)
*   **Weeks 1–4 (Pure GP Exploration & ARD Capping):** The initial strategy relied on an automated Gaussian Process (anisotropic Matérn kernel) featuring ARD, driven by Upper Confidence Bound (UCB, $\kappa \in [3.0, 4.0]$) on $F_1, F_2, F_4, F_6, F_7, F_8$ and Expected Improvement (EI) on $F_3, F_5$, plus function-specific `length_scale_bounds` caps in Week 3 to prevent premature dimension pruning.
*   **Weeks 5–9 (Transition to EI, Trust-Region Expansion & Diagnostic Overrides):** Shifted all functions to Expected Improvement ($\xi$ annealed from `0.1` to `0.0001`), expanded adaptive trust regions (scaled to $1.5\times$ the top-3 point spread) across $F_4, F_6, F_7, F_8$, and introduced targeted human-in-the-loop coordinate probing (probing past the negative anomaly on $F_1$, 1D coordinate stepping on $F_6$, and boundary splicing on $F_8$).
*   **Weeks 10–12 (Hybrid NN Validation & Analytical Exploitation):** Integrated the PyTorch `64x64` ReLU Neural Network surrogate with active-subspace pruning ($F_1$) to cross-audit GP proposals, removed the legacy `[1e-6, 0.999999]` coordinate clamp, and deployed analytical 2D log-Gaussian calibration ($F_1$) and Quadratic RSM ($F_8$)—delivering **5 all-time records in Week 12**.
*   **Week 13 (Final-Round Exploitation Refinement):** Codified `compute_final_refinement` in [`src/bbo_optimize.py`](../src/bbo_optimize.py), resolving one-sided sampling bias on $F_1$, inactive-dimension EI drift on $F_2/F_3$, global penalty-cliff variance inflation on $F_4$, noise-induced length-scale collapse on $F_6$, and restoring $d_3 = 0.305066$ inside a local GP trust region on $F_7$ to achieve a massive finale breakthrough (`2.385595`).

---

## Performance

### Evaluation Methodology & Analysed Datasets
*   **How performance is measured:** Primary performance is evaluated by the **cumulative maximum oracle output** ($y^\ast = \max_{t \in \{0..13\}} y_t$) achieved on each of the 8 hidden functions over 13 sequential weekly queries, supplemented by GP posterior calibration ($\mu \pm \sigma$), normalised exploitation index ($\mu / \sigma$), and leave-one-out cross-validation ($\text{LOO RMSE}$).
*   **Datasets analysed:** Evaluated across the 8 Imperial College BBO Capstone datasets (`data/accumulated/function_1` through `function_8`, comprising 23, 23, 28, 43, 29, 33, 43, and 53 points respectively; 275 total oracle evaluations).

### Final Campaign Metrics (Weeks 0–13)

| Function | Dim ($d$) | Samples ($N$) | Baseline Best (W0) | Week 13 Return | **All-Time Best ($y^\ast$)** | Best Week | Total Gain ($\text{W0} \to \text{Best}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **F1 ($F_1$)** (Sparse Needle) | 2D | 23 | `0.000000` | `1.99999995` | **`1.99999999`** | W12 | $\mathbf{+2.000000}$ |
| **F2 ($F_2$)** (Noisy Sim) | 2D | 23 | `0.611205` | `0.482805` | **`0.721870`** | W12 | $\mathbf{+0.110665}$ |
| **F3 ($F_3$)** (Drug Discovery) | 3D | 28 | `-0.034835` | `-0.017169` | **`-0.006045`** | W11 | $\mathbf{+0.028790}$ |
| **F4 ($F_4$)** (Warehouse) | 4D | 43 | `-4.025542` | `0.641291` | **`0.679981`** | W12 | $\mathbf{+4.705523}$ |
| **F5 ($F_5$)** (Chemical Yield) | 4D | 29 | `1088.8596` | **`8662.482500`** | **`8662.482500`** | W12 & W13 | $\mathbf{+7573.6229}$ |
| **F6 ($F_6$)** (Cake Recipe) | 5D | 33 | `-0.714265` | `-0.242430` | **`-0.090586`** | W10 | $\mathbf{+0.623679}$ |
| **F7 ($F_7$)** (ML Hyperparams) | 6D | 43 | `1.364968` | **`2.385595`** | **`2.385595`** | **W13** | $\mathbf{+1.020627}$ |
| **F8 ($F_8$)** (ML Hyperparams) | 8D | 53 | `9.598884` | `9.983502` | **`9.987569`** | W12 | $\mathbf{+0.388685}$ |

### Performance Summary Graphs (Weeks 1–13)

![13-Week Optimization Trajectory Dashboard](../results/week_13/progress/trajectory_dashboard_week13.png)

![13-Week Normalized GP Metrics Dashboard](../results/week_13/progress/metrics_dashboard_normalized_week13.png)

---

## Limitations (Assumptions and Limitations)
*   **Stationarity & Homoscedasticity Assumptions:** Standard Matérn GPs assume that surface roughness ($\boldsymbol{\ell}$) and noise variance ($\sigma_n^2$) are stationary across $[0, 1]^d$. On sharply non-stationary functions like $F_1$ (where 98% of the space is flat zero and the signal is a narrow Gaussian spike), an unpruned global GP over-smooths the spike unless restricted to a local active bounding box.
*   **Variance Inflation & Kriging Nugget Shrinkage:** When fitting a global GP across extreme outer penalty cliffs (such as $F_4$, which drops to $-32.63$), global output variance is inflated by $>100\times$ relative to the summit plateau, while tightly clustered summit points drive predictive uncertainty $\sigma(\mathbf{x})$ toward zero, causing raw EI to stall.
*   **Small-Sample Local Overconfidence:** Conversely, when pruning a GP to a very small local subset ($n = 9$ positive points on $F_4$), leave-one-out variance increases ($\text{LOO RMSE} = 0.197$) and normal-tail improvement probabilities become overconfident outside a tight radius ($\|\Delta \mathbf{x}\|_2 \le 0.01$).
*   **Piecewise-Linear Neural Extrapolation:** Unregularised ReLU Neural Networks extrapolate linearly outside the convex hull of training data, which can overshoot interior quadratic peaks ($F_1, F_8$) if not constrained by explicit trust-region boxes.

---

## Trade-offs
*   **Short-Term Score vs. Long-Term Information Gain (De-Confounding):** Updating all $d$ coordinates simultaneously maximises expected single-step gain when the surrogate is well-calibrated, but confounds individual variable effects when the score drops (as seen on $F_6$ in Week 13 when stepping $d_2, d_4, d_5$ simultaneously to the trust-box corner). Conversely, One-Factor-At-A-Time (OFAT) stepping sacrifices single-week upside to isolate exact partial derivatives, which proved critical on $F_1, F_6,$ and $F_7$.
*   **ARD Length-Scale Capping vs. Nuisance Filtering:** Tight upper bounds on ARD length scales (`length_scale_bounds=(1e-2, 0.5)`) force the GP to explore subtle dimensions early on, but if kept too tight in later weeks, they induce artificial EI drift along genuinely flat nuisance dimensions ($d_2$ on $F_2$, $d_4$ on $F_3$, $d_2/d_6$ on $F_8$).
*   **Global Coverage vs. Local Precision:** Active-learning clustering produces sub-millimetre precision near the global peak ($10^{-6}$ coordinate accuracy on $F_1$), at the trade-off of leaving low-performing regions of the hypercube unmapped.

---

## Ethical Considerations
*   **How does transparency support reproducibility and real-world adaptation?** 
    In sparse black-box optimization, automated surrogates frequently exhibit silent failure modes (such as $\ell_5$ collapsing to $0.0127$ on $F_6$ when fitting noisy observations without a `WhiteKernel`, or sample-density pullback on $F_8$). Transparency requires that human-in-the-loop interventions—data pruning masks, trust-region bounds, inactive-dimension locks, and random seeds—are codified directly in [`src/bbo_optimize.py`](../src/bbo_optimize.py) (`compute_final_refinement`) and audited against leave-one-out and multi-surrogate checks rather than presented as opaque intuition.
