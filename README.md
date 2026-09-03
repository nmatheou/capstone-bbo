# Capstone Black-Box Optimization (BBO)

This repository contains the codebase and weekly submissions for a Black-Box Optimization capstone project. The objective is to find the global maxima of 8 unknown, continuous black-box functions with varying dimensionalities (ranging from 2D to 8D) over a multi-week period.

## GitHub Documentation Strategy

The repository is anchored by a comprehensive README designed to serve as an executive summary for facilitators, peers, and future employers. The documentation explicitly bridges theory and code, detailing the mathematical rationale behind the GP kernel selection and the strategic shift from UCB to EI. Embedded visualizations of the surrogate model's convergence and the stabilization of length scales over the weekly submissions act as tangible proof of the model's sample efficiency.

For a deep dive into the methodology and to reproduce the results, please refer to the specific documentation files:

*   [**Architecture Flow**](docs/architecture.md): A detailed breakdown of the Oracle -> Data -> GP -> Acquisition -> Submission pipeline.
*   [**Reproducibility Guide**](docs/reproducibility.md): Instructions on how to set up the environment and run the optimizer for any given week.
*   [**Diagnostics & Overrides**](docs/diagnostics_and_overrides.md): Details the logic behind CV thresholds, EI vs UCB switching, and how manual overrides (like triangulation and boundary locking) were applied to combat GP limitations.
*   [**Challenges & Future Improvements**](docs/challenges_and_improvements.md): Covers issues like length-scale saturation in high dimensions and plans for multi-fidelity BO.

## Technical Justification & Literature

The core technical justification for the current approach relies on utilizing a Gaussian Process (GP) as a probabilistic surrogate model. GPs natively quantify predictive uncertainty, which is mathematically essential for calculating acquisition functions to efficiently balance the exploration-exploitation trade-off in limited-query, continuous environments.

The foundational text *Gaussian Processes for Machine Learning* by Rasmussen and Williams underpins the choice of covariance functions (kernels) and the optimization of length scales.

Furthermore, research on acquisition functions provides the theoretical backing for efficiently climbing the "hill" toward the global maximum. Specifically, the literature on the Upper Confidence Bound (UCB) strategy supports the initial explorative phase, while the Expected Improvement (EI) algorithm—introduced by Jones et al.—justifies the transition to a more exploitative approach, including adjustments to the xi parameter to force greedy gains during later submissions.

## Frameworks and Libraries

The implementation relies heavily on the core Python data science stack, specifically NumPy, SciPy, and Pandas.

*   **NumPy and SciPy:** Explicitly chosen for their highly optimized linear algebra capabilities, which are absolutely critical for handling the intensive matrix inversion operations required by GPs. Using these foundational libraries provides fine-grained control over algorithm optimization and matrix operations, ensuring lightweight execution compared to the heavy overhead of deep learning frameworks like PyTorch or TensorFlow. SciPy specifically drives the acquisition phase, using scipy.optimize.minimize (multi-start L-BFGS-B) to find optimal query points, and scipy.stats.qmc (Latin Hypercube Sampling) to seed the optimizer.
*   **scikit-learn:** Standardized utilities are incorporated for data scaling (StandardScaler), baseline evaluation, and the core surrogate model (GaussianProcessRegressor) using the Matérn kernel.
*   **Custom diagnostic scripts:** Used for length-scale tracking, coefficient-of-variation analysis, and irrelevant-dimension detection.

**Trade-offs considered:** GPs scale poorly in high dimensions. I mitigate this mathematically using UCB exploration and irrelevant-dimension detection, and logically using manual domain reasoning and dimension locking (e.g., forcing linear slopes to boundaries). I considered PyTorch and TensorFlow, but they introduce unnecessary complexity for this project and reduce auditability. My choices prioritise stability and interpretability.

## Repository Structure

My repository is structured around the weekly workflow of the BBO capstone. At the core is my optimisation engine (src/bbo_optimize.py) and supporting diagnostic scripts. All historical oracle data is stored in the data/ directory, which accumulates baseline samples and weekly returns.

Weekly outputs are separated into two areas:
*   esults/: Contains the suggested queries generated for each week, alongside plots and progress trackers.
*   submissions/: Stores the files I actually submitted to the oracle.

This structure reflects the iterative nature of the project: new data arrives weekly, I update the optimiser, review diagnostics, and prepare the submission.

*   src/: Contains all Python source code.
*   configs/: Stores week-specific parameters (kernel settings, acquisition strategies, kappa/xi values).
*   data/: Contains all raw and accumulated Oracle returns.
*   docs/: Contains methodology, architecture documentation, and reproducibility guides.
*   logs/: Contains automated run logs capturing GP hyperparameters, acquisition values, and diagnostics.

## Workflow

1. Update data/accumulated/ with the latest oracle returns.
2. Modify configs/hyperparameters.json if algorithmic tuning is needed.
3. Run python src/bbo_optimize.py --week <N> to generate raw suggestions and update logs.
4. Review diagnostic logs and manual override strategies.
5. Package the suggestions into submissions/week_<N>/.

## Future Refinements

To continue refining the strategy, consulting the Black-Box Optimization Benchmarking (BBOB) framework would provide rigorous, standardized continuous functions to stress-test the model's performance.

Reviewing recent literature on high-dimensional Bayesian Optimization, such as Trust Region Bayesian Optimization (TuRBO), would offer pathways to scale the algorithm for more complex, multi-dimensional landscapes.
