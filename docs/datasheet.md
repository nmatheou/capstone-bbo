# Datasheet for Black-Box Optimization (BBO) Datasets

This document provides a description of the sequential datasets collected as part of the Imperial College Capstone project on Black-Box Optimization.

## 1. Motivation

**Purpose of Dataset:**
This dataset was constructed in an iterative manner to tackle a tightly constrained Black-Box Optimization (BBO) problem. As the governing analytical expressions of the systems were fully hidden (black-box), the only viable approach to characterising the structure of the search spaces was through empirical data generation.

**Supporting Task:**
The dataset is used to train surrogate models, including Gaussian Processes and shallow neural networks, in order to optimise a scalar reward over eight different functions. These functions simulate real-world optimisation tasks such as radioactive source localisation, noisy physical simulations, drug discovery, warehouse logistics, chemical yield maximisation, recipe optimisation, and machine learning hyperparameter tuning.

## 2. Composition

**Contents of Dataset:**
The dataset consists of eight separate subsets (Functions 1 to 8). Each subset includes pairs of continuous input coordinates $\mathbf{x} \in [0, 1]^d$, where the dimensionality $d$ varies from 2 to 8, along with a corresponding scalar output $y \in \mathbb{R}$.

**Size and Format:**
*   **Size:** The dataset is highly sparse. Across the completed 13-week campaign, each function contains 23 to 53 data points (23, 23, 28, 43, 29, 33, 43, and 53 points for $F_1$–$F_8$ respectively, totalling 275 evaluations), comprising an initial blind sampling batch (10 to 40 points) plus 13 sequential weekly oracle evaluations.
*   **Format:** Raw weekly oracle returns are stored as `.txt` files under `data/week_<1..13>_returns/` (`inputs.txt` and `outputs.txt`) and consolidated into both NumPy `.npy` binaries and human-readable `.csv` files under `data/accumulated/function_<1..8>/` (`initial_inputs.npy`, `initial_outputs.npy`, `initial_inputs.csv`, `initial_outputs.csv`).

**Data Gaps:**
Significant gaps exist due to the inherent sparsity of sampling in high-dimensional spaces. For example, in 8D Function 8, 53 points represent an extremely small fraction of the overall domain. The dataset also reflects intentional active-learning sampling bias, prioritising exploitation of high-value basins while leaving low-performing regions sparsely sampled. In Function 1 (2D Sparse Needle), 15 of the 23 observations are near zero ($\sim 10^{-16}$ to $10^{-35}$), with all 8 positive observations concentrated inside $[0.6285, 0.6636] \times [0.6285, 0.6640]$.

## 3. Collection Process

**Query Process:**
Data was generated sequentially across the completed 13-week campaign (13 oracle evaluation rounds across Weeks 1–13), with one query per function per week selected via active learning.

**Query Strategy:**
The strategy evolved from pure surrogate-based optimisation to a hybrid Human-in-the-Loop method. Initially, a Gaussian Process model with an anisotropic Matérn kernel and UCB/EI acquisition functions guided sampling. Over time, small-sample GP limitations became apparent, including over-smoothing of sharp needles ($F_1$), global variance inflation and Kriging nugget shrinkage ($F_4$), noise-induced length-scale collapse ($F_6$), and sample-density pullback in 8D ($F_8$). To resolve these, I introduced PyTorch 64-64 ReLU Neural Network surrogates, analytical log-Gaussian and Quadratic Response Surface (RSM) derivations, positive-basin pruning, `WhiteKernel` noise regularisation, and explicit trust-region constraints.

**Collection Timeframe:**
All data was collected during the 13-week sequential submission campaign for the Imperial College Capstone project.

## 4. Pre-processing and Uses

**Transformations Applied:**
Before surrogate training, target values $y$ are standardised to zero mean and unit variance (`normalize_y=True` in `GaussianProcessRegressor` and explicit $z$-score normalisation in `train_and_optimize_nn`) to stabilise Cholesky factorisation and neural network gradient descent.

**Data Pruning:**
*   **Function 1:** Because the global dataset is dominated by zero-response observations, neural network training is restricted to the active bounding box $[0.62, 0.67] \times [0.62, 0.67]$ (8 positive summit points across the completed campaign).
*   **Function 4 (Week 13 Final Refinement):** Because outer penalty cliffs (down to $-32.63$) inflate global output variance by $114\times$ relative to the summit cluster, the Local GP ($\nu=1.5$) in `compute_final_refinement` prunes the dataset to the positive summit observations ($y > 0$).

**Use Case:**
The dataset is intended for local surrogate modelling and sequential active-learning benchmarking aimed at identifying global maxima under strict query budgets.

**Limitations:**
The dataset is not suitable for learning globally calibrated response surfaces across the full $[0, 1]^d$ hypercube. Due to strong active-learning clustering around discovered peaks, global surrogates trained on this data over-weight high-density clusters and have high epistemic uncertainty in unexplored corners.

## 5. Distribution and Maintenance

**Data Storage:**
The dataset is stored locally within the project repository under `data/accumulated/`, `data/week_<1..13>_returns/`, `results/week_<1..13>/`, and `submissions/week_<1..13>/`.

**Terms of Use:**
Use of the dataset is restricted to academic evaluation in accordance with Imperial College Capstone project guidelines.

**Maintenance:**
The dataset is maintained by the student researcher through the conclusion of the Week 13 final capstone submission.
