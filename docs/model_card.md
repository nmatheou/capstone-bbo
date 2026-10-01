# Model Card: Human-in-the-Loop Hybrid Surrogate Optimizer

## Overview
*   **Name of approach:** Human-in-the-Loop Hybrid Surrogate Optimizer (HitL-HSO)
*   **Type:** Sequential Black-Box Optimization (BBO) ensemble, combining Gaussian Process Regression (GPR), Shallow Neural Networks (NN), and Human Heuristics.
*   **Version:** 1.0 (Week 10 Capstone Submission)

## Intended Use
*   **What tasks is it suitable for?** 
    This model is designed for high-cost, zero-derivative, severely budget-constrained black-box optimization tasks (e.g., physical sensor placement, experimental drug discovery, expensive ML hyperparameter tuning) where data sparsity is extreme ($<50$ points) and dimensions range from 2D to 8D.
*   **What use cases should be avoided?** 
    It is completely unsuitable for generating highly accurate *global* topological maps, as the acquisition function is hyper-focused on local exploitation. Furthermore, it should be avoided in fully autonomous, non-stationary continuous deployment systems, as the architecture fundamentally relies on manual human interventions to prevent model collapse.

## Details
*   **Explain your strategy across the ten rounds, including the techniques you used and how your approach evolved.** 
    *   **Weeks 1–4 (Pure GP Exploration):** The initial strategy relied purely on an automated Gaussian Process (Matérn $\nu=2.5$ kernel) featuring Automatic Relevance Determination (ARD), driven by Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition functions.
    *   **Weeks 5–8 (The Human Intervention):** The strategy evolved rapidly as the GP began exhibiting severe mathematical failure modes. High epistemic uncertainty caused the GP to hallucinate non-existent peaks on the absolute boundaries (the "Valley of Death"). Furthermore, the Matérn kernel artificially smoothed sharp discontinuities (F1 Sparse Needle). Human domain logic (e.g., the "Radiation Shield" triangulation hypothesis) was injected to override the GP and safely map gradients.
    *   **Weeks 9–10 (Hybrid NN Validation):** The approach matured into a robust hybrid system. To correct ARD length-scale saturation (which was blinding the GP to active dimensions), we executed calculated "sacrifices"—deliberately querying extreme boundary errors to force the covariance matrix to recalibrate. We also integrated Shallow ReLU Neural Networks to audit the GP. By applying strict "Data Pruning" (micro-bounding boxes) to eliminate background noise, the NN was able to natively validate complex linear boundaries and topological gradients that the standard GP smoothed over.

## Performance
*   **Summarize your results across the eight functions. What metrics did you use?** 
    *   **Metrics:** The sole metric is the target deterministic scalar output ($Y$) representing the optimization objective, measured against the severely constrained query budget. 
    *   **Results (As of Week 10):**
        *   **F1 (Sparse Needle):** Escaped the $0.0$ baseline noise floor, mapping the steep cliff to `0.012+`.
        *   **F2, F3, F4, F6, F7:** Successfully unblinded the ARD models and perfectly bracketed the absolute ceilings of local topological pockets, routinely scoring highly competitive optimums.
        *   **F5:** Completely solved (locked to absolute theoretical upper bounds).
        *   **F8 (8D Hyperparameters):** Reached a score of `9.958`, resting at $99.58\%$ of the theoretical absolute maximum of $10.0$ via the "Splicing Strategy".

## Assumptions and Limitations
*   **What assumptions underlie your strategy?** 
    A core assumption is that the contextual background descriptions of the functions reveal theoretical bounds (e.g., a cap of $0.0$ for F3/F6 and $10.0$ for F8). We also assumed the standard Gaussian Process operates under homoscedasticity (constant noise variance) and stationarity, which informed our decision to override it when tackling heteroscedastic environments (sharp cliffs).
*   **What are its constraints or failure modes?** 
    The strategy is heavily dependent on human judgement. The primary failure mode is placing too much confidence in an assumed theoretical maximum; if a bound is assumed to be lower than reality, the system will prematurely exploit a local peak and fail to explore better global regions. Conversely, the automated GP's failure mode is ARD length scale saturation in sparse regimes, causing it to dismiss relevant dimensions as "dead" axes.

## Ethical Considerations
*   **How does transparency support reproducibility and real-world adaptation?** 
    In sparse black-box environments, automated surrogate models frequently fail. Transparency demands that human interventions—such as data pruning rules, manual coordinate locks (splicing), and random seed initializations—are explicitly documented rather than disguised as "intuition." By transparently documenting exactly *where* and *why* the Gaussian Process hallucinated, we ensure that real-world practitioners adapting this strategy for high-stakes physical environments (e.g., autonomous chemical synthesis) do not blindly trust acquisition scores in sparse regions without implementing secondary validation (like our Shallow NN checks).
