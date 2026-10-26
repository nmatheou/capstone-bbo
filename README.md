# Capstone Black-Box Optimization (BBO)

This repository contains the complete codebase, datasets, interactive presentation notebook, governance documentation, and weekly submissions (Weeks 1–13) for the Imperial College Black-Box Optimization (BBO) capstone project. The objective is to find the global maxima of 8 unknown, continuous black-box functions with varying dimensionalities (ranging from 2D to 8D) over a 13-week sequential query campaign.

---

## Non-Technical Summary (General Audience)

Many real-world challenges—such as discovering effective drug formulations, tuning machine-learning systems, or maximising industrial chemical yields—require finding the best settings for a system whose internal formula is hidden and where every test is slow or expensive. In this 13-week project, I optimised eight unknown "black-box" functions ranging from two to eight input variables, allowed only one test per function each week. I built a hybrid decision engine combining probabilistic Gaussian Process models—which estimate both expected performance and uncertainty—with shallow neural networks and human-in-the-loop diagnostics. Across 13 rounds, all eight functions achieved substantial gains, including hitting the exact theoretical peak on two benchmarks.

---

## Quick Navigation & Repository Index

All project files—including code, data, presentation materials, and governance documents—are stored directly in this repository and linked below:

### 1. Code Presentation & Interactive Walkthrough
*   [**Jupyter Notebook Presentation (`bbo_capstone_presentation.ipynb`)**](bbo_capstone_presentation.ipynb): Interactive notebook presenting the methodology, well-commented surrogate implementation, 13-week progression table, and embedded visual dashboards.
*   [**Main Optimization Engine (`src/bbo_optimize.py`)**](src/bbo_optimize.py): End-to-end CLI pipeline implementing anisotropic Matérn Gaussian Processes, UCB/EI acquisition functions, PyTorch `64x64` ReLU Neural Network verification (`train_and_optimize_nn`), and human-in-the-loop refinements (`compute_final_refinement`).
*   [**Diagnostic & Tracking Scripts (`src/`)**](src/):
    *   [`src/progress_tracker.py`](src/progress_tracker.py): Generates weekly trajectory and GP mean/std dashboards.
    *   [`src/plot_normalized_metrics.py`](src/plot_normalized_metrics.py): Plots normalized exploitation index ($\mu / \sigma$) and exploration ratios.
    *   [`src/length_scale_tracker.py`](src/length_scale_tracker.py): Tracks Automatic Relevance Determination (ARD) kernel length scales across weeks.
    *   [`src/pca_analysis.py`](src/pca_analysis.py): Performs Principal Component Analysis (PCA) on active-basin trajectories.
*   [**Hyperparameter Configuration (`configs/hyperparameters.json`)**](configs/hyperparameters.json): Externalised random seeds, UCB $\kappa$, and EI $\xi$ settings.

### 2. Governance, Transparency & Technical Documentation (`docs/`)
*   [**Datasheet (`docs/datasheet.md`)**](docs/datasheet.md): Complete Module 21 Datasheet detailing dataset motivation, composition ($n = 275$ total evaluations across 8 functions), active-learning collection process, pre-processing/pruning, and limitations.
*   [**Model Card (`docs/model_card.md`)**](docs/model_card.md): Complete Module 21 Model Card documenting the Human-in-the-Loop Hybrid Surrogate Optimizer (HitL-HSO), its intended use, 13-week evolution, final performance, assumptions, failure modes, and ethical/transparency considerations.
*   [**Architecture Flow (`docs/architecture.md`)**](docs/architecture.md): Detailed breakdown of the Oracle $\to$ Data $\to$ GP / NN $\to$ Acquisition / Refinement $\to$ Submission pipeline.
*   [**Reproducibility Guide (`docs/reproducibility.md`)**](docs/reproducibility.md): Environment setup (`numpy`, `scipy`, `scikit-learn`, `matplotlib`, `torch`) and exact CLI commands to reproduce any week (`--week 1` through `--week 13`).
*   [**Diagnostics & Overrides (`docs/diagnostics_and_overrides.md`)**](docs/diagnostics_and_overrides.md): Explains CV thresholds, UCB-to-EI transitions, and human-in-the-loop overrides (local positive-basin pruning, `WhiteKernel` noise regularisation, trust-region bounds, and inactive-dimension anchoring).
*   [**Challenges & Future Improvements (`docs/challenges_and_improvements.md`)**](docs/challenges_and_improvements.md): Analysis of ARD length-scale saturation, Kriging nugget shrinkage, and pathways toward Trust Region Bayesian Optimization (TuRBO).
*   **Weekly Submission Logic & Post-Mortem Reports:**
    *   [**Weeks 1–11 Historical Submission Logic**](docs/submission_logic_w1_to_w11.md)
    *   [**Week 12 Submission Logic**](docs/week_12_submission_logic.md) & [**Week 12 Oracle Analysis**](docs/week_12_analysis.md)
    *   [**Week 13 Final Submission Logic**](docs/week_13_submission_logic.md) & [**Week 13 Final Analysis & 13-Week Post-Mortem**](docs/week_13_analysis.md)

### 3. Datasets, Weekly Submissions & Visual Results
*   [**Accumulated Datasets (`data/accumulated/`)**](data/accumulated/): Complete `.npy` and `.csv` input/output datasets through Week 13 for [`function_1`](data/accumulated/function_1/) through [`function_8`](data/accumulated/function_8/).
*   [**Weekly Oracle Returns (`data/week_<1..13>_returns/`)**](data/): Raw weekly `inputs.txt` and `outputs.txt` files returned by the evaluation oracle ([`week_1`](data/week_1_returns/) to [`week_13`](data/week_13_returns/)).
*   [**Weekly Submissions Archive (`submissions/`)**](submissions/): Formatted coordinates and scripts submitted for [`week_1`](submissions/week_1/) through [`week_13`](submissions/week_13/).
*   [**Visual Dashboards & Surface Plots (`results/`)**](results/): Weekly progress dashboards and 2D acquisition plots from [`week_1`](results/week_1/) to [`week_13`](results/week_13/).
*   [**Automated Run Logs (`logs/run_logs.txt`)**](logs/run_logs.txt): Timestamped log of GP kernel fits, ARD length scales, and NN predictions.

---

## Final 13-Week Campaign Results

Across the 13-week campaign, **all 8 functions achieved substantial improvements over their initial baselines**, with 7 of the 8 functions setting their all-time records in Weeks 11–13 (and all 8 between Weeks 10 and 13):

| Function | Dim | Total Samples ($N$) | Initial Best (W0) | Final W13 Return | **All-Time Maximum** | Best Week | Campaign Gain ($\text{W0} \to \text{Max}$) | Key Method Driving Peak |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$F_1$** (Sparse Needle) | 2D | 23 | `0.000000` | `1.99999995` | **`1.99999999`** | W12 | **$+2.000000$** | 2D Log-Gaussian Secant Calibration |
| **$F_2$** (Noisy Simulation) | 2D | 23 | `0.611205` | `0.482805` | **`0.721870`** | W12 | **$+0.110665$** | 1D GP Ridge Mean + `WhiteKernel` ($d_2=0.387$) |
| **$F_3$** (Drug Discovery) | 3D | 28 | `-0.034835` | `-0.017169` | **`-0.006045`** | W11 | **$+0.028790$** | Boundary Pocket Triangulation ($d_4$ Ignored) |
| **$F_4$** (Warehouse Logistics) | 4D | 43 | `-4.025542` | `0.641291` | **`0.679981`** | W12 | **$+4.705523$** | Matérn $\nu=1.5$ + Positive-Basin Trust Region |
| **$F_5$** (Chemical Yield) | 4D | 29 | `1088.8596` | **`8662.482500`** | **`8662.482500`** | W12 & W13 | **$+7573.6229$** | Unclamped `[1, 1, 1, 1]` Boundary Exploitation |
| **$F_6$** (Cake Recipe) | 5D | 33 | `-0.714265` | `-0.242430` | **`-0.090586`** | W10 | **$+0.623679$** | Controlled 1D Coordinate Descent (`d3, d5`) |
| **$F_7$** (ML Hyperparameters) | 6D | 43 | `1.364968` | **`2.385595`** | **`2.385595`** | **W13** | **$+1.020627$** | Trust-Region GP Mean Synthesis ($d_1=0$) |
| **$F_8$** (ML Hyperparameters) | 8D | 53 | `9.598884` | `9.983502` | **`9.987569`** | W12 | **$+0.388685$** | 2nd-Order Quadratic RSM + PyTorch NN |

### Final Campaign Dashboards (Week 13)

![13-Week Optimization Trajectory Dashboard](results/week_13/progress/trajectory_dashboard_week13.png)

![13-Week Normalized GP Metrics Dashboard](results/week_13/progress/metrics_dashboard_normalized_week13.png)

---

## Data Storage & Large Dataset Policy

*   **Empirical BBO Datasets (`data/`):** Because each function was queried sequentially once per week across 13 rounds, the complete empirical dataset comprises 275 total observations across all 8 functions ($23 \le n \le 53$ per function, $< 50\text{ KB}$ total). These compact datasets are stored directly in the repository in both NumPy binary (`.npy`) and human-readable (`.csv`) formats under [`data/accumulated/`](data/accumulated/), alongside raw weekly oracle files in [`data/week_1_returns/`](data/week_1_returns/) through [`data/week_13_returns/`](data/week_13_returns/).
*   **Large Candidate Search Grids (Generated On-Demand):** During each optimization run, the pipeline evaluates dense multi-dimensional candidate meshes ($10,000$-point Latin Hypercube Sampling matrices per function plus multi-start L-BFGS-B/Nelder-Mead trajectories). In accordance with GitHub best practices, these large intermediate evaluation matrices are **not hosted directly on GitHub**. Instead, they are deterministically generated in memory on demand via `scipy.stats.qmc.LatinHypercube(d, seed=42)` when running [`src/bbo_optimize.py`](src/bbo_optimize.py) with the seeds specified in [`configs/hyperparameters.json`](configs/hyperparameters.json).
*   **External Benchmark Reference:** The functional archetypes in this project draw inspiration from standard continuous black-box optimization suites such as the [**COCO / Black-Box Optimization Benchmarking (BBOB) Suite**](https://numbbo.github.io/coco/testsuites/bbob).

---

## Technical Justification & Literature

The core technical justification relies on a Gaussian Process (GP) as the primary probabilistic surrogate model, paired with a secondary 2-layer (`64x64` ReLU) Neural Network surrogate and analytical response-surface models for late-stage cross-validation. GPs natively quantify predictive uncertainty ($\mu(\mathbf{x}), \sigma(\mathbf{x})$), which is essential for acquisition functions that balance exploration and exploitation under severe sample sparsity ($23 \le n \le 53$).

The foundational text *Gaussian Processes for Machine Learning* by Rasmussen and Williams underpins the choice of anisotropic Matérn covariance functions ($\nu \in \{1.5, 2.5\}$) and Automatic Relevance Determination (ARD) length-scale optimization. Research on acquisition functions supports the two-stage campaign: Upper Confidence Bound (UCB, Srinivas et al.) drives early exploration, while Expected Improvement (EI, Jones et al.) and trust-region posterior-mean refinement drive final-round exploitation.

## Frameworks and Libraries

The implementation relies on the Python scientific computing stack alongside PyTorch for neural surrogate validation:

*   **NumPy and SciPy:** Used for array manipulation, Cholesky-based GP linear algebra, Latin Hypercube Sampling (`scipy.stats.qmc`), and multi-start acquisition / trust-region optimization (`scipy.optimize.minimize` with `L-BFGS-B` and `Nelder-Mead`).
*   **scikit-learn:** Provides the core `GaussianProcessRegressor` implementation with composite `ConstantKernel`, anisotropic `Matern`, and `WhiteKernel` covariance structures, as well as `SVC` for active-subspace classification on $F_1$.
*   **PyTorch (`torch`):** Powers the secondary `64x64` ReLU Neural Network surrogate (`train_and_optimize_nn` in [`src/bbo_optimize.py`](src/bbo_optimize.py)), trained with Adam and optimized via L-BFGS gradient ascent from the empirical best point to cross-check GP proposals and capture directional momentum in higher dimensions ($F_8$).
*   **Matplotlib:** Generates 2D acquisition surface slices, historical trajectory dashboards, normalized progress metrics, and ARD length-scale evolution plots.

**Trade-offs considered:** Standard global GPs suffer from variance inflation, Kriging nugget shrinkage across tight summit clusters, and ARD length-scale saturation on nuisance axes. I mitigate these failure modes by combining GP posterior diagnostics with positive-basin pruning ($F_1, F_4$), explicit `WhiteKernel` noise modeling ($F_2, F_6$), inactive-dimension anchoring ($F_2, F_3$), and Neural Network / Quadratic RSM cross-validation ($F_8$).

---

## Reproducible Workflow

1. Place the latest oracle returns in `data/week_<N-1>_returns/inputs.txt` and `outputs.txt`.
2. Modify [`configs/hyperparameters.json`](configs/hyperparameters.json) if algorithmic tuning is needed.
3. Run `python src/bbo_optimize.py --week <N>` (e.g., `python src/bbo_optimize.py --week 13`) to append new returns (dedup-guarded), fit GP and NN surrogates, apply final-round refinements, and generate `results/week_<N>/suggestions.txt` and `submissions/week_<N>/suggestions.txt`.
4. Run diagnostic trackers (`python src/progress_tracker.py --week 13`, `python src/plot_normalized_metrics.py --week 13`, `python src/length_scale_tracker.py --week 13`) and review the interactive walkthrough in [`bbo_capstone_presentation.ipynb`](bbo_capstone_presentation.ipynb).
