# BBO Optimization Architecture

## Core Pipeline

1. **Oracle Input**
   - Data arrives as .npy arrays inside data/week_{N-1}_returns/.
   - Historical accumulated data is kept in data/accumulated/.

2. **Surrogate Modeling (scikit-learn)**
   - Engine: GaussianProcessRegressor
   - Kernel: Matern(nu=2.5) — provides a strong balance of smoothness and flexibility.
   - Preprocessing: StandardScaler stabilizes the GP against dramatically varying output scales (e.g., F5).

3. **Acquisition Phase (SciPy)**
   - Seeds: scipy.stats.qmc.LatinHypercube creates an initial space-filling design.
   - Optimizer: scipy.optimize.minimize(method="L-BFGS-B") performs a multi-start optimization to find the mathematical peak of the acquisition surface.
   - Strategy: UCB (exploration) vs EI (exploitation) dictated by week and function diagnostics.

4. **Output Generation**
   - Logs written to logs/run_logs.txt.
   - Suggested coordinates written to esults/week_{N}/suggestions.txt.
   - 2D Acquisition surfaces saved to esults/week_{N}/plots/.
