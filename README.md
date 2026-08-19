# Capstone Black-Box Optimization (BBO)

This repository contains the codebase and weekly submissions for a Black-Box Optimization capstone project. The objective is to find the global maxima of 8 unknown, continuous black-box functions with varying dimensionalities (ranging from 2D to 8D) over a multi-week period.

## Repository Structure

My repository is structured around the weekly workflow of the BBO capstone. At the core is my optimisation engine (src/bbo_optimize.py) and supporting diagnostic scripts. All historical oracle data is stored in the data/ directory, which accumulates baseline samples and weekly returns.

Weekly outputs are separated into two areas:
*   esults/: Contains the suggested queries generated for each week, alongside plots and progress trackers.
*   submissions/: Stores the files I actually submitted to the oracle.

This structure reflects the iterative nature of the project: new data arrives weekly, I update the optimiser, review diagnostics, and prepare the submission.

*   src/: Contains all Python source code.
*   configs/: Stores week-specific parameters (kernel settings, acquisition strategies, $\kappa$/$\xi$ values).
*   data/: Contains all raw and accumulated Oracle returns.
*   docs/: Contains methodology, architecture documentation, and reproducibility guides.
*   logs/: Contains automated run logs capturing GP hyperparameters, acquisition values, and diagnostics.

## Coding Libraries and Packages

My approach relies on a lightweight, stable, and reproducible stack:
*   **scikit-learn:** Used for the core surrogate (GaussianProcessRegressor), the Matérn kernel ($\nu = 2.5$), and StandardScaler for target normalisation.
*   **SciPy:** The mathematical engine driving the acquisition phase, specifically using scipy.optimize.minimize (multi-start L-BFGS-B) to find optimal query points, and scipy.stats.qmc (Latin Hypercube Sampling) to seed the optimizer.
*   **Custom diagnostic scripts:** Used for length-scale tracking, coefficient-of-variation analysis, and irrelevant-dimension detection.

**Why these choices are appropriate:** Gaussian Processes are well-suited to low-sample black-box optimisation because they provide the reliable uncertainty estimates required by acquisition functions. The Matérn kernel offers a strong balance between smoothness and flexibility across varied landscapes. scikit-learn and SciPy are deterministic and well-tested, ensuring full reproducibility.

**Trade-offs considered:** GPs scale poorly in high dimensions. I mitigate this mathematically using UCB exploration and irrelevant-dimension detection, and logically using manual domain reasoning and dimension locking (e.g., forcing linear slopes to boundaries). I considered PyTorch and TensorFlow, but they introduce unnecessary complexity for this project and reduce auditability. My choices prioritise stability and interpretability.

## Documentation Reference

For a deep dive into the methodology and to reproduce the results, please refer to the specific documentation files:

*   [**Architecture Flow**](docs/architecture.md): A detailed breakdown of the Oracle $\rightarrow$ Data $\rightarrow$ GP $\rightarrow$ Acquisition $\rightarrow$ Submission pipeline.
*   [**Reproducibility Guide**](docs/reproducibility.md): Instructions on how to set up the environment and run the optimizer for any given week.
*   [**Diagnostics & Overrides**](docs/diagnostics_and_overrides.md): Details the logic behind CV thresholds, EI vs UCB switching, and how manual overrides (like triangulation and boundary locking) were applied to combat GP limitations.
*   [**Challenges & Future Improvements**](docs/challenges_and_improvements.md): Covers issues like length-scale saturation in high dimensions and plans for multi-fidelity BO.

## Workflow

1. Update data/accumulated/ with the latest oracle returns.
2. Modify configs/hyperparameters.json if algorithmic tuning is needed.
3. Run python src/bbo_optimize.py --week <N> to generate raw suggestions and update logs.
4. Review diagnostic logs and manual override strategies.
5. Package the suggestions into submissions/week_<N>/.
