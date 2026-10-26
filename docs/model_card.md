# Model Card: Human-in-the-Loop Hybrid Surrogate Optimizer

## Overview
*   **Name of approach:** Human-in-the-Loop Hybrid Surrogate Optimizer (HitL-HSO)
*   **Type:** Sequential Black-Box Optimization (BBO) ensemble, combining Gaussian Process Regression (GPR), Shallow Neural Networks (PyTorch 64-64 ReLU MLP), Response Surface Methodology (RSM), and Human-in-the-Loop Diagnostics.
*   **Version:** 1.3 (Final Week 13 Capstone Submission)

## Intended Use
*   **What tasks is it suitable for?** 
    This model is designed for high-cost, zero-derivative, severely budget-constrained black-box optimization tasks (e.g., physical sensor placement, experimental drug discovery, expensive ML hyperparameter tuning) where data sparsity is extreme ($23 \le n \le 53$ points across 13 rounds, 275 total evaluations) and dimensions range from 2D to 8D.
*   **What use cases should be avoided?** 
    It is unsuitable for generating globally calibrated topological maps across the entire domain, as the late-stage acquisition strategy is hyper-focused on local summit exploitation. Furthermore, it should be avoided in fully autonomous, non-stationary continuous deployment systems without human oversight, as small-sample surrogates require diagnostic intervention to prevent boundary and covariance failure modes.

## Details
*   **Strategy Evolution Across Thirteen Rounds:** 
    *   **Weeks 1–4 (Pure GP Exploration):** The initial strategy relied on an automated Gaussian Process (anisotropic Matérn $\nu=2.5$ kernel) featuring Automatic Relevance Determination (ARD), driven by Upper Confidence Bound (UCB) and Expected Improvement (EI) acquisition functions.
    *   **Weeks 5–8 (Diagnostic Human Intervention):** The strategy evolved as the GP exhibited classic small-sample failure modes: high epistemic uncertainty at domain boundaries pulled unconstrained optimizers into severe penalty cliffs, while smooth Matérn kernels over-smoothed localized needles ($F_1$). Domain-informed triangulation and trust-region constraints were introduced to stabilize climbing trajectories.
    *   **Weeks 9–12 (Hybrid NN Validation & Analytical Exploitation):** I integrated a seeded 2-layer (64-64 ReLU) PyTorch Neural Network surrogate with active-subspace pruning ($F_1$) to cross-audit GP proposals, alongside analytical response-surface methods—most notably an exact 2D log-Gaussian derivative derivation on $F_1$ and Quadratic Response Surface Methodology (RSM) with ARD nuisance relaxation ($d_5, d_8$) on $F_8$—which delivered **5 new all-time records in Week 12**.
    *   **Week 13 (Final-Round Exploitation Refinement):** For the final submission (`compute_final_refinement` in `src/bbo_optimize.py`), I diagnosed and corrected four late-stage surrogate distortions: (1) one-sided northeast sampling bias on $F_1$ (resolved via a $+10^{-6}$ diagonal step to `0.628542-0.628542`), (2) inactive-dimension EI drift on $F_2$ and $F_3$ (resolved by locking proven ridge/pocket coordinates $d_2 = 0.387$ on $F_2$ and $(d_1, d_2) = (0.95, 0.40)$ on $F_3$), (3) global variance inflation and Kriging nugget shrinkage on $F_4$ and $F_6$ (resolved via positive-basin Local GP pruning on $F_4$ and `WhiteKernel` noise regularization on $F_6$), and (4) sample-density pullback toward old clusters on $F_8$ (resolved by selecting the 64-64 ReLU NN local gradient-ascent refinement around the Week 12 record).

## Performance
*   **Metrics:** The primary metric is the observed scalar oracle output ($y^* = \max_{t} y_t$) under the strict sequential query budget, supplemented by GP posterior mean ($\mu$), posterior standard deviation ($\sigma$), and leave-one-out cross-validation diagnostics.
*   **Results (Final All-Time Best Scores after Week 13 Returns):**
    *   **$F_1$ (2D Sparse Needle):** **`1.99999999`** (All-time record in Week 12 at `0.628541-0.628541`, within $1.35 \times 10^{-8}$ of the $2.0$ ceiling; Week 13 at `0.628542-0.628542` returned `1.99999995`, bracketing the exact continuous center at $0.6285408\dots$).
    *   **$F_2$ (2D Noisy Sim):** **`0.721870`** (All-time record in Week 12 at `0.689600-0.387000`, surpassing the Week 3 high of `0.641577` by $+0.080293$; Week 13 at `0.688100-0.387000` returned a downward stochastic draw of `0.482805`).
    *   **$F_3$ (3D Drug Discovery):** **`-0.006045`** (All-time record in Week 11 at `0.950000-0.400000-0.438611`; Week 13 on the left shoulder of Needle #2 at `0.950000-0.400000-0.434766` returned `-0.017169`, recovering $+0.014340$ over Week 12's `-0.031509`).
    *   **$F_4$ (4D Warehouse Logistics):** **`0.679981`** (All-time record in Week 12 at `0.407968-0.405671-0.359691-0.420080`; Week 13 Local GP peak at `0.405930-0.406376-0.352329-0.422444` returned `0.641291`, the second-highest score of the 43-point campaign).
    *   **$F_5$ (4D Chemical Yield):** **`8662.482500`** (All-time record matched in both Week 12 and Week 13 at the exact hyper-corner `1.000000-1.000000-1.000000-1.000000`).
    *   **$F_6$ (5D Cake Composite):** **`-0.090586`** (All-time record in Week 10 at `0.450000-0.300000-0.662500-0.750000-0.087500`; Week 13 trust-box corner test returned `-0.242430`).
    *   **$F_7$ (6D ML Hyperparameters):** **`2.385595`** (**New all-time record in Week 13** at `0.000000-0.156068-0.305066-0.315590-0.335798-0.766358`, surging **$+0.184279$** above Week 11's `2.201316` and **$+0.248399$** above Week 12's `2.137196` by restoring $d_3 = 0.305066$ alongside the refined $(d_2, d_4, d_5, d_6)$ coordinates).
    *   **$F_8$ (8D ML Hyperparameters):** **`9.987569`** (All-time record in Week 12 via Quadratic RSM active-subspace shift; Week 13 NN refinement at `0.157119-0.175371-0.120105-0.090651-0.893686-0.492256-0.191545-0.504012` returned `9.983502`, the second-highest score in history and within $0.0019$ of the Matérn-2.5 GP prediction `9.985399`).

## Assumptions and Limitations
*   **What assumptions underlie my strategy?** 
    I assume that local smoothness holds within compact trust regions around high-performing basins and that noise on stochastic functions ($F_2, F_6$) can be approximated as homoscedastic Gaussian white noise (`WhiteKernel`).
*   **What are its constraints or failure modes?** 
    Global GPs fitted across extreme penalty cliffs ($F_4$) suffer from variance inflation and nugget shrinkage, while local GPs fitted on very small subsets ($n = 9$ on $F_4$) exhibit elevated leave-one-out variance ($\text{LOO RMSE} = 0.197$) and overconfident normal-tail probabilities. Similarly, unregularized ReLU Neural Networks can extrapolate linearly past theoretical ceilings ($F_1, F_8$) or toward domain boundaries ($F_7$). Bounding step sizes within explicit trust regions is therefore essential.

## Ethical Considerations
*   **How does transparency support reproducibility and real-world adaptation?** 
    In sparse black-box optimization, automated surrogates frequently exhibit silent failure modes (such as $\ell_5$ collapsing to $0.0127$ on $F_6$ when fitting noisy observations without a `WhiteKernel`, or sample-density pullback on $F_8$). Transparency requires that human-in-the-loop interventions—data pruning masks, trust-region bounds, inactive-dimension locks, and random seeds—are codified directly in `src/bbo_optimize.py` (`compute_final_refinement`) and audited against leave-one-out and multi-surrogate checks rather than presented as black-box intuition.
