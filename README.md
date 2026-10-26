# Capstone Black-Box Optimization (BBO)

This repository contains my codebase and weekly submissions for the Imperial College Black-Box Optimization (BBO) capstone project. The objective is to find the global maxima of 8 unknown, continuous black-box functions with varying dimensionalities (ranging from 2D to 8D) over a 13-week sequential query campaign.

## GitHub Documentation Strategy

The repository is anchored by this README to serve as an executive summary of the methodology, architecture, and empirical progression across Weeks 1–13. The documentation explicitly bridges theory and code, detailing the mathematical rationale behind the Gaussian Process (GP) kernel selection, the transition from Upper Confidence Bound (UCB) exploration to Expected Improvement (EI) and posterior-mean exploitation, and the integration of a secondary Shallow Neural Network (PyTorch) surrogate and analytical Response Surface Methodology (RSM).

For a deep dive into the methodology and to reproduce the results, please refer to the specific documentation files:

*   [**Datasheet**](docs/datasheet.md): Details on why the dataset was developed, what it contains, how it is processed, collected, and maintained through Week 13.
*   [**Model Card**](docs/model_card.md): Details on the hybrid surrogate optimizer's purpose, design, limitations, performance across all 13 rounds, usage, and ethical considerations.
*   [**Architecture Flow**](docs/architecture.md): A detailed breakdown of the Oracle $\to$ Data $\to$ GP / NN $\to$ Acquisition / Refinement $\to$ Submission pipeline.
*   [**Reproducibility Guide**](docs/reproducibility.md): Instructions on how to set up the environment (`numpy`, `scipy`, `scikit-learn`, `matplotlib`, `torch`) and run the optimizer for any given week, including `--week 13`.
*   [**Diagnostics & Overrides**](docs/diagnostics_and_overrides.md): Details the logic behind CV thresholds, EI vs. UCB switching, and how human-in-the-loop overrides (local positive-basin pruning, `WhiteKernel` noise regularization, trust-region bounds, and inactive-dimension anchoring) were applied to resolve GP failure modes.
*   [**Challenges & Future Improvements**](docs/challenges_and_improvements.md): Covers issues like ARD length-scale saturation in high dimensions, Kriging nugget shrinkage, and pathways toward Trust Region Bayesian Optimization (TuRBO).
*   [**Historical Submission Logic (W1–W11)**](docs/submission_logic_w1_to_w11.md), [**Week 12 Analysis**](docs/week_12_analysis.md), [**Week 12 Submission Logic**](docs/week_12_submission_logic.md), [**Week 13 Final Submission Logic**](docs/week_13_submission_logic.md), and [**Week 13 Final Analysis & Post-Mortem**](docs/week_13_analysis.md).

## Technical Justification & Literature

The core technical justification relies on a Gaussian Process (GP) as the primary probabilistic surrogate model, paired with a secondary 2-layer (64-64 ReLU) Neural Network surrogate and analytical response-surface models for late-stage cross-validation. GPs natively quantify predictive uncertainty ($\mu(\mathbf{x}), \sigma(\mathbf{x})$), which is essential for acquisition functions that balance exploration and exploitation under severe sample sparsity ($22 \le n \le 52$).

The foundational text *Gaussian Processes for Machine Learning* by Rasmussen and Williams underpins the choice of anisotropic Matérn covariance functions ($\nu \in \{1.5, 2.5\}$) and Automatic Relevance Determination (ARD) length-scale optimization. Research on acquisition functions supports the two-stage campaign: Upper Confidence Bound (UCB, Srinivas et al.) drives early exploration, while Expected Improvement (EI, Jones et al.) and trust-region posterior-mean refinement drive final-round exploitation.

## Frameworks and Libraries

The implementation relies on the Python scientific computing stack alongside PyTorch for neural surrogate validation:

*   **NumPy and SciPy:** Used for array manipulation, Cholesky-based GP linear algebra, Latin Hypercube Sampling (`scipy.stats.qmc`), and multi-start acquisition / trust-region optimization (`scipy.optimize.minimize` with `L-BFGS-B` and `Nelder-Mead`).
*   **scikit-learn:** Provides the core `GaussianProcessRegressor` implementation with composite `ConstantKernel`, anisotropic `Matern`, and `WhiteKernel` covariance structures, as well as `SVC` for active-subspace classification on $F_1$.
*   **PyTorch (`torch`):** Powers the secondary 64-64 ReLU Neural Network surrogate (`train_and_optimize_nn` in `src/bbo_optimize.py`), trained with Adam and optimized via L-BFGS gradient ascent from the empirical best point to cross-check GP proposals and capture directional momentum in higher dimensions ($F_8$).
*   **Matplotlib:** Generates 2D acquisition surface slices, historical trajectory dashboards, normalized progress metrics, and ARD length-scale evolution plots.

**Trade-offs considered:** Standard global GPs suffer from variance inflation, Kriging nugget shrinkage across tight summit clusters, and ARD length-scale saturation on nuisance axes. I mitigate these failure modes by combining GP posterior diagnostics with positive-basin pruning ($F_1, F_4$), explicit `WhiteKernel` noise modeling ($F_2, F_6$), inactive-dimension anchoring ($F_2, F_3$), and Neural Network / Quadratic RSM cross-validation ($F_8$).

## Repository Structure

My repository is structured around the weekly workflow of the BBO capstone. At the core is my optimization engine (`src/bbo_optimize.py`) and supporting diagnostic scripts (`src/progress_tracker.py`, `src/plot_normalized_metrics.py`, `src/length_scale_tracker.py`). All historical oracle data is stored in the `data/` directory, which accumulates baseline samples and weekly returns.

Weekly outputs are separated into two areas:
*   `results/`: Contains the suggested queries generated for each week, alongside acquisition surface plots and progress dashboards.
*   `submissions/`: Stores the exact formatted coordinate files (`suggestions.txt`) and rationale documents submitted to the oracle each week.

Directory overview:
*   `src/`: Contains all Python source code for optimization and progress tracking.
*   `configs/`: Stores hyperparameter settings (`configs/hyperparameters.json`).
*   `data/`: Contains raw weekly oracle returns (`data/week_<N>_returns/`) and consolidated datasets (`data/accumulated/function_<1..8>/`).
*   `docs/`: Contains methodology, architecture documentation, model card, datasheet, and weekly submission logic.
*   `logs/`: Contains automated run logs (`logs/run_logs.txt`) capturing GP hyperparameters, NN predictions, and final refinement coordinates.

## Workflow

1. Place the latest oracle returns in `data/week_<N-1>_returns/inputs.txt` and `outputs.txt`.
2. Modify `configs/hyperparameters.json` if algorithmic tuning is needed.
3. Run `python src/bbo_optimize.py --week <N>` (e.g., `python src/bbo_optimize.py --week 13`) to append new returns (dedup-guarded), fit GP and NN surrogates, apply final-round refinements, and generate `results/week_<N>/suggestions.txt` and `submissions/week_<N>/suggestions.txt`.
4. Run diagnostic trackers (`src/progress_tracker.py`, `src/plot_normalized_metrics.py`, `src/length_scale_tracker.py`) and record the decision rationale in `docs/week_<N>_submission_logic.md`.

## Future Refinements

To continue refining the strategy beyond the 13-week capstone budget, consulting the Black-Box Optimization Benchmarking (BBOB) framework would provide standardized continuous functions to stress-test the hybrid pipeline, and adopting automated Trust Region Bayesian Optimization (TuRBO) would formalize the local positive-basin GP strategy deployed on $F_4, F_6,$ and $F_7$.
