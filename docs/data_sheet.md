# Datasheet for Black-Box Optimization (BBO) Datasets (Functions 1–8)

This datasheet documents the datasets, optimisation decisions, weekly learning, and reasoning across all eight functions ($F_1$ to $F_8$) in the Imperial College Black-Box Optimization (BBO) Capstone project, integrating both the **Module 25 Final Project Datasheet Template** and the **Module 21 Datasheet Sections**.

---

## Function Overview

### 1. Which function does this datasheet describe? (Functions 1–8)
This datasheet covers the complete suite of eight continuous black-box functions (**Functions 1 to 8**, denoted $F_1$ to $F_8$) optimised in parallel across the 13-week campaign:
*   **F1:** 2D Contamination / Radiation Source Detection ("Sparse Needle")
*   **F2:** 2D Noisy Physical / Dual-Peak Simulation
*   **F3:** 3D Drug Discovery / Adverse Reaction Minimisation
*   **F4:** 4D Warehouse Logistics / Dynamic Placement
*   **F5:** 4D Industrial Chemical Yield Maximisation
*   **F6:** 5D Cake Recipe / Multi-Component Culinary Formulation
*   **F7:** 6D Machine Learning Hyperparameter Tuning
*   **F8:** 8D High-Dimensional Machine Learning Hyperparameter Tuning

### 2. What real-world scenario does each function simulate?
*   **F1 (2D):** Detecting a highly localised two-dimensional contamination or radiation plume across a wide geographical grid where 98% of background readings register zero.
*   **F2 (2D):** Calibrating a noisy two-parameter physical or mystery ML model with stochastic output variance and an invariant second parameter.
*   **F3 (3D):** Optimising a three-compound drug formulation to minimise adverse side effects ($y \le 0$, optimum near $0.0$).
*   **F4 (4D):** Optimising 4D warehouse resource allocation where poor configurations trigger severe logistical bottlenecks (steep penalty cliffs down to $-32.63$).
*   **F5 (4D):** Maximising chemical reaction yield across four continuous process variables (temperature, pressure, catalyst concentration, residence time).
*   **F6 (5D):** Balancing a 5-ingredient cake recipe evaluated by a composite negative loss score ($y \le 0$, optimum near $0.0$).
*   **F7 (6D):** Tuning six continuous hyperparameters of a machine-learning model to maximise validation performance.
*   **F8 (8D):** Tuning eight continuous hyperparameters of a larger machine-learning system containing both primary active parameters and redundant nuisance parameters.

### 3. What is the dimensionality of the input?
All input domains are bounded unit hypercubes $\mathbf{x} \in [0, 1]^d$:
*   **2D:** $F_1, F_2$
*   **3D:** $F_3$
*   **4D:** $F_4, F_5$
*   **5D:** $F_6$
*   **6D:** $F_7$
*   **8D:** $F_8$

### 4. How many initial data points were provided?
At Week 0 (prior to sequential querying), the initial datasets contained **171 total points** across the 8 functions:

| Function | Initial Input Shape ($\mathbf{X}_0$) | Initial Output Shape ($\mathbf{y}_0$) | Final Input Shape ($\mathbf{X}_{13}$) | Final Output Shape ($\mathbf{y}_{13}$) |
| :--- | :---: | :---: | :---: | :---: |
| **F1** | `(10, 2)` | `(10,)` | `(23, 2)` | `(23,)` |
| **F2** | `(10, 2)` | `(10,)` | `(23, 2)` | `(23,)` |
| **F3** | `(15, 3)` | `(15,)` | `(28, 3)` | `(28,)` |
| **F4** | `(30, 4)` | `(30,)` | `(43, 4)` | `(43,)` |
| **F5** | `(16, 4)` | `(16,)` | `(29, 4)` | `(29,)` |
| **F6** | `(20, 5)` | `(20,)` | `(33, 5)` | `(33,)` |
| **F7** | `(30, 6)` | `(30,)` | `(43, 6)` | `(43,)` |
| **F8** | `(40, 8)` | `(40,)` | `(53, 8)` | `(53,)` |

### 5. What does the output represent?
In all eight tasks, the output $y \in \mathbb{R}$ represents a **scalar objective score to be maximised**:
*   **F1:** Signal intensity ($y \in [-0.0036, 2.0]$, theoretical maximum $= 2.0$).
*   **F2:** Noisy model performance score (observed range $[-0.066, 0.722]$).
*   **F3:** Negated adverse side-effect severity ($y \le 0$, observed range $[-0.398, -0.0060]$, ideal ceiling $= 0.0$).
*   **F4:** Net warehouse efficiency score (observed range $[-32.63, +0.680]$).
*   **F5:** Chemical yield output (observed range $[0.11, 8662.4825]$).
*   **F6:** Negated recipe defect score ($y \le 0$, observed range $[-2.57, -0.0906]$, ideal ceiling $= 0.0$).
*   **F7:** Validation accuracy/score metric (observed range $[0.0027, 2.3856]$).
*   **F8:** Validation performance metric (observed range $[5.59, 9.9876]$, theoretical ceiling $= 10.0$).

---

## Nature of the Data

### 1. Structure of the initial dataset
Each function's dataset is stored as a pair of NumPy arrays (`initial_inputs.npy` of shape $(n, d)$ in `float64` and `initial_outputs.npy` of shape $(n,)$ in `float64`, mirrored as `.csv` files in [`data/accumulated/`](../data/accumulated/)). Initial points were distributed quasi-randomly across $[0, 1]^d$, leaving vast regions of the interior and boundaries unsampled.

### 2. How the dataset evolves as new queries are added weekly
Across 13 sequential rounds (Weeks 1–13), **1 new coordinate per function per week** (+13 points per function, +104 points overall, bringing the total from 171 to 275 evaluations) was appended via `src/bbo_optimize.py`. Because queries were selected via active learning, the distribution shifted from broad space-filling exploration in Weeks 1–4 to dense, high-value local clusters around discovered summits in Weeks 5–13 (creating intentional active-learning sampling bias).

### 3. Does the function include noise or randomness?
*   **Deterministic Functions ($F_1, F_5, F_7, F_8$):** Repeated or adjacent queries at $10^{-6}$ separation confirmed zero stochastic noise. Most notably, querying `(1, 1, 1, 1)` on $F_5$ in both Week 12 and Week 13 returned the exact same value (`8662.4825000000008`), and querying `(0.628541, 0.628541)` vs. `(0.628542, 0.628542)` on $F_1$ matched an exact deterministic 2D Gaussian to 8 significant figures.
*   **Noisy / Stochastic Functions ($F_2, F_6$, and mild boundary noise on $F_3$):**
    *   **F2:** Repeated queries along the $d_1 \approx 0.69 \text{ to } 0.70$ ridge produced stochastic fluctuations with standard deviation $\sigma_{\text{obs}} \approx 0.094$ (e.g., `0.6416` in W3 vs. `0.5567` in W11 at $\Delta = 0.0028$; `0.7219` in W12 vs. `0.4828` in W13 at $\Delta = 0.0015$).
    *   **F6:** Exhibited evaluation noise ($\sigma_{\text{obs}} \approx 0.06 \text{ to } 0.09$) that caused an unregularised GP's $\ell_5$ length scale to collapse to `0.0127` until we added a `WhiteKernel`.

### 4. Unimodal, multimodal, noisy, or smooth classification
*   **F1 (2D):** **Non-stationary unimodal needle** with a negative "radiation shield" trough (`-0.0036` at `[0.65, 0.68]`) bordering a narrow isotropic Gaussian peak (`2.0` at `[0.628541, 0.628541]`) and flat zero everywhere else.
*   **F2 (2D):** **Noisy ridge landscape**—strongly curved along $d_1$ (peak at $d_1 \approx 0.69$) and invariant along $d_2$ ($\ell_2 = 100.0$).
*   **F3 (3D):** **Multimodal pocket landscape**—contains at least two distinct local peaks along $d_3$ (`d3 ≈ 0.439` and `d3 ≈ 0.483`) inside the high $d_1$ boundary pocket (`d1 = 0.95, d2 = 0.40`).
*   **F4 (4D):** **Rugged, sharp-crested interior peak** ($\nu=1.5$) around `[0.408, 0.406, 0.360, 0.420]` surrounded by severe non-linear penalty cliffs (`-32.63`).
*   **F5 (4D):** **Smooth, monotonically increasing unimodal function** peaking at the upper boundary corner `[1, 1, 1, 1]`.
*   **F6 (5D):** **Noisy interior basin** centered near `[0.45, 0.30, 0.6625, 0.75, 0.0875]`.
*   **F7 (6D):** **Smooth coupled ridge** peaking on the $d_1 = 0.0$ boundary plane with strong positive synergy across $(d_2, d_3, d_4, d_5, d_6)$.
*   **F8 (8D):** **Smooth concave quadratic bowl** dominated by 4 active dimensions ($d_1, d_3, d_4, d_7$), 2 weak boundary-sloped dimensions ($d_5, d_8$), and 2 flat nuisance dimensions ($d_2, d_6$).

---

## Your Optimisation Strategy

### 1. Which optimisation method(s) did you use?
I designed a **Human-in-the-Loop Hybrid Surrogate Optimizer** combining:
1.  **Bayesian Optimisation with Anisotropic Matérn Gaussian Processes (`scikit-learn`):** Equipped with Automatic Relevance Determination (ARD), `WhiteKernel` noise modelling, and UCB / EI acquisition functions maximized via Latin Hypercube Sampling (`n = 10,000, seed = 42`) + `L-BFGS-B`.
2.  **Adaptive Trust-Region Constraints:** Bounding candidate searches around the top-3 historical points (`tr_width = np.maximum(0.3, spread * 1.5)`), tightened in Week 13.
3.  **Shallow Neural Network Surrogate (`PyTorch`):** A 2-layer (`64x64` ReLU) MLP trained with Adam and optimised via L-BFGS input gradient ascent (`train_and_optimize_nn`).
4.  **Analytical Response Surface Methodology (RSM) & Controlled Coordinate Probing:** Exact 2D log-Gaussian secant calibration ($F_1$), 2nd-order Quadratic RSM ($F_8$), and One-Factor-At-A-Time (OFAT) coordinate descent ($F_6$).

### 2. Why did you choose this method for each function?
*   **F1:** Standard GPs failed on the 15 flat-zero points; combining an `SVC` active-subspace filter, `[0.62, 0.67]^2` data pruning, and analytical 2D log-Gaussian calibration matched the exact exponential peak geometry.
*   **F2 & $F_6$:** Added `WhiteKernel` noise regularisation and 1D coordinate locking / OFAT stepping so the optimiser wouldn't chase random noise draws.
*   **F3:** Capped ARD length scales (`2.0`) and locked $(d_1, d_2) = (0.95, 0.40)$ to isolate the sharp $d_3$ needle ($\ell_3 = 0.01$).
*   **F4:** Selected a rougher Matérn $\nu=1.5$ kernel from Week 1 and pruned out negative penalty cliffs ($y > 0$) inside a tight local trust region.
*   **F5:** Used EI with unclamped `[0.0, 1.0]` boundaries so the optimiser could climb directly into the `[1, 1, 1, 1]` hyper-corner.
*   **F7 & $F_8$:** Combined ARD length-scale filtering, adaptive trust regions, Quadratic RSM, and PyTorch NN gradient ascent to navigate 6D and 8D spaces without falling into boundary uncertainty traps.

### 3. How did you balance exploration and exploitation?
*   **Weeks 1–4 (Exploration):** Used global Latin Hypercube Sampling and UCB ($\kappa \in [3.0, 4.0]$) with capped ARD length scales to force exploration of under-sampled dimensions.
*   **Weeks 5–9 (Transition to Local Exploitation):** Shifted all functions to Expected Improvement ($\xi$ reduced from `0.1` $\to$ `0.01` $\to$ `0.0001`) inside adaptive trust regions around the top-3 points.
*   **Weeks 10–13 (Precision Summit Exploitation & Directed Ridge Stepping):** Used posterior-mean maximisation ($\mu(\mathbf{x})$), analytical response surfaces, and neural network gradient ascent within tight local radii ($\pm 0.015 \text{ to } 0.05$).

### 4. Did your strategy change over the weeks? Why?
Yes, substantially:
*   **Week 3:** Added ARD `length_scale_bounds` caps after noticing length scales saturating to `100.0`.
*   **Weeks 5–6:** Shifted from UCB to EI, relaxed the $F_2$ cap so $d_2$ could drop out naturally, and intervened manually on $F_1$ after 5 weeks of `0.0000` returns revealed GP stationarity blindness.
*   **Weeks 7–9:** Expanded adaptive trust regions to $F_4$ and $F_6$ (breaking the $F_4$ 8-week plateau in Week 9) and deployed boundary splicing on $F_8$.
*   **Weeks 10–13:** Integrated the PyTorch `64x64` ReLU NN surrogate, removed the `[1e-6, 0.999999]` coordinate clamp, and introduced [`compute_final_refinement`](../src/bbo_optimize.py) for local basin pruning and analytical summit calibration.

---

## Data Handling and Preprocessing

### 1. Did you rescale or normalise inputs and outputs? Why or why not?
*   **Inputs ($\mathbf{x}$):** Already bounded in $[0, 1]^d$, so global input rescaling was unnecessary; however, for Quadratic RSM fitting on $F_8$, inputs were centered relative to the incumbent best ($\Delta \mathbf{x} = \mathbf{x} - \mathbf{x}_{\text{best}}$) to improve numerical conditioning.
*   **Outputs ($y$):** Standardized to zero mean and unit variance (`normalize_y=True` in `GaussianProcessRegressor` and explicit $(y - \mu_y)/\sigma_y$ standardisation in `train_and_optimize_nn`) because output scales varied by 6 orders of magnitude across functions (from $10^{-3}$ on $F_1/F_3$ to $8662.48$ on $F_5$).

### 2. Did you train any surrogate models?
Yes:
1.  **Anisotropic Matérn Gaussian Process Regressors** (`scikit-learn`) on all 8 functions.
2.  **RBF Support Vector Classifier (`SVC`)** on $F_1$ (Weeks 1–9) to classify active vs. zero regions.
3.  **2-Layer (`64x64` ReLU) Neural Network Regressors** (`PyTorch`) on all 8 functions (Weeks 10–13).
4.  **Analytical Log-Gaussian ($F_1$) and 2nd-Order Ridge Quadratic Response Surface ($F_8$) Models** (Weeks 11–13).

### 3. Preprocessing and tuning required by the surrogates
*   **Kernel Choice:** `ConstantKernel(1.0) * Matern(length_scale=ones(d), nu=1.5 or 2.5)` plus `WhiteKernel` on $F_2$ and $F_6$.
*   **Hyperparameter Tuning:** Multi-start L-BFGS-B (`n_restarts_optimizer=10–25`) for GP log-marginal likelihood; Adam (`lr=0.005, weight_decay=1e-4, epochs=800`) for PyTorch MLP training.

### 4. Did you handle outliers or unusual data points?
Yes—handling extreme values and structural anomalies was decisive:
*   **F1 Negative Anomaly & Zero Plane:** Point #14 (`[0.65, 0.68] -> -0.003606`) was the only non-zero clue among 15 zeros; rather than discarding it, we diagnosed it as a "radiation shield" trough adjacent to the true peak, and pruned the 15 outer zero points (`[0.62, 0.67]^2` filter) once the positive peak was found so the NN and log-Gaussian solver wouldn't be flattened.
*   **F4 Penalty Cliffs:** Outer points dropped as low as `-32.63`, inflating global GP variance by $114\times$ relative to the positive summit cluster. In Week 13, we pruned $F_4$ to its $y > 0$ positive basin for local GP refinement.

---

## Weekly Iteration and Learning

### 1. How did new data points change your understanding of the function landscape?
Each weekly return acted as a structural diagnostic:
*   On **F1**, Weeks 6–10 (`0.0026 -> 0.0154`) revealed a steep exponential slope heading southwest toward `(0.6285, 0.6285)`, proving the peak was an isotropic 2D Gaussian rather than a broad plateau.
*   On **F7**, Week 12's drop (`2.2013 -> 2.1372`) when $d_3$ was lowered to `0.154` proved that $d_3 \approx 0.305$ was a critical ridge crest, directly inspiring our Week 13 synthesis (`2.385595`).
*   On **F8**, Week 12's jump (`9.9616 -> 9.9876`) proved that relaxing $d_5$ and $d_8$ off the `1.0` boundary while shifting $d_1, d_3, d_4, d_7$ captured the true interior bowl vertex.

### 2. Did you encounter local optima? How did you detect them?
*   **F3 Dual Needles:** Detected two distinct optima along $d_3$ (`d3 = 0.483` yielding `-0.0117` in W9 vs. `d3 = 0.439` yielding `-0.0060` in W11) separated by a deeper trough at `d3 = 0.460` (`-0.0149` in W10) by plotting 1D cross-sections along $d_3$.
*   **F8 Boundary Trap (`9.9616`):** Weeks 7–10 plateaued around `9.96` with $d_5=1.0, d_8=1.0$; fitting a 2nd-order Quadratic RSM revealed that the true unconstrained vertex lay slightly inside the boundary (`d5=0.885, d8=0.500`), unlocking `9.9876`.

### 3. Which queried inputs were most informative and why?
*   **F1 Week 6 (`0.657424-0.663992` $\to 0.002561$):** First positive signal after 5 weeks of zeros, confirming the active basin location.
*   **F1 Week 11 (`0.635000-0.635000` $\to 0.709490$):** $46\times$ breakthrough that provided the third non-collinear point needed to solve the exact 2D Gaussian equations for Week 12 (`2.0000`).
*   **F6 Weeks 8–10 OFAT steps (`d3 = 0.60 -> 0.65 -> 0.6625`):** Cleanly mapped the parabolic apex on $d_3$ and $d_5$ to hit `-0.090586`.
*   **F7 Week 13 (`0.000000-0.156068-0.305066-0.315590-0.335798-0.766358` $\to 2.385595$):** Confirmed super-additive synergy between the Week 11 $d_3$ crest and Week 12 $(d_2, d_4, d_5, d_6)$ shifts.

### 4. If you restarted, what would you do differently?
1.  **Interrogate Initial Anomalies in Week 1:** Investigate unusual baseline points (like the $F_1$ single negative reading `-0.0036`) immediately rather than waiting until Week 5.
2.  **Bracket Both Flanks of Every Slope Early:** Design early exploratory queries to bracket both sides of a peak rather than clustering on one flank, enabling exact quadratic/Gaussian interpolation several weeks sooner.
3.  **Automate Regime-Shift & Stationarity Audits:** Build automated LOO residual and variance-inflation checks into the pipeline from Week 1.

---

## Performance and Results

### 1 & 2. Best output values achieved and exact input vectors

| Function | Best Output Achieved ($y^\ast$) | Best Week | Exact Input Vector $\mathbf{x}^\ast$ Producing Best Output | Confidence Near Global Max |
| :--- | :---: | :---: | :--- | :---: |
| **F1 (2D)** | **`1.99999999`** | W12 | `0.628541-0.628541` | **Virtually 100%** (Within $1.35 \times 10^{-8}$ of exact $2.0$ ceiling) |
| **F2 (2D)** | **`0.721870`** | W12 | `0.689600-0.387000` | **High** (At exact $d_1 \approx 0.69$ ridge peak + $+1.2\sigma$ noise draw) |
| **F3 (3D)** | **`-0.006045`** | W11 | `0.950000-0.400000-0.438611` | **High** (Bracketed Needle #2 crest within $0.006$ of $0.0$ ceiling) |
| **F4 (4D)** | **`0.679981`** | W12 | `0.407968-0.405671-0.359691-0.420080` | **High** (Confirmed by W13 `#2` score `0.6413` at $\Delta = 0.008$) |
| **F5 (4D)** | **`8662.482500`** | W12 & W13 | `1.000000-1.000000-1.000000-1.000000` | **100%** (Exact deterministic upper hyper-corner maximum) |
| **F6 (5D)** | **`-0.090586`** | W10 | `0.450000-0.300000-0.662500-0.750000-0.087500` | **High** (Within $0.09$ of $0.0$ ceiling on noisy 5D surface) |
| **F7 (6D)** | **`2.385595`** | **W13** | `0.000000-0.156068-0.305066-0.315590-0.335798-0.766358` | **Very High** ($+1.021$ gain over W0, $+0.184$ jump in W13) |
| **F8 (8D)** | **`9.987569`** | W12 | `0.145000-0.180000-0.132000-0.098000-0.885000-0.505000-0.192000-0.500000` | **99.9%** (Within $0.0124$ of $10.0$ theoretical ceiling) |

### 3. How confident are you that these are near the global maximum, and why?
*   **Exact Theoretical Ceilings Reached ($F_1, F_5, F_8$):** On $F_1$ (`1.99999999 / 2.0`), $F_5$ (`8662.4825` at `[1,1,1,1]`), and $F_8$ (`9.9876 / 10.0`), both analytical derivations and Week 13 bracketing queries confirm we are within $0.00\% \text{ to } 0.12\%$ of the true analytical optimum.
*   **Bracketed Summit Ridges ($F_2, F_3, F_4, F_6, F_7$):** On $F_3$ and $F_6$, scores reached `-0.0060` and `-0.0906` against a non-positive ceiling ($y \le 0$), while on $F_4$ (`0.6800`) and $F_7$ (`2.3856`), tight local trust-region probes confirmed high-altitude summit plateaus with low GP posterior standard deviation ($\sigma \le 0.043$).

### 4. Did your results align with expectations based on the problem descriptions?
Yes: the $F_1$ "radiation plume" matched a localized 2D Gaussian point source; $F_2$ and $F_6$ exhibited the expected experimental noise; $F_3$ and $F_6$ behaved as negative-loss penalty functions ($y \le 0$); the $F_5$ chemical yield scaled monotonically to the maximum boundary corner; and $F_7/F_8$ exhibited classic ML hyperparameter sensitivity where a subset of parameters ($d_1, d_3, d_4, d_7$ on $F_8$) dominated performance while redundant parameters had negligible effect.

---

## Ethical, Practical and General Considerations

### 1. How does this black-box optimisation task relate to real-world applications?
Sequential BBO under a strict 13-query budget directly mirrors high-cost experimental workflows—such as wet-lab drug synthesis ($F_3$), industrial chemical reactor trials ($F_5$), clinical protocol design, and large-scale neural architecture tuning ($F_7, F_8$)—where each evaluation requires days of compute or physical laboratory resources.

### 2. What limitations arise from the synthetic nature of the functions?
Synthetic benchmarks are defined over clean, box-constrained hypercubes $[0, 1]^d$ without missing values, sensor drift, delayed feedback, or safety-critical hard constraints (where querying an unsafe coordinate could damage physical equipment rather than merely returning `-32.63` as on $F_4$).

### 3. Would your strategy scale to more serious or more expensive problems? Why or why not?
*   **Where it scales well:** The hybrid GP + ARD + Adaptive Trust Region + Local Response Surface pipeline scales effectively to moderate-dimensional ($d \le 15$), expensive-evaluation problems ($N < 200$) where sample efficiency and interpretability are paramount.
*   **Where adaptation is needed:** For ultra-high-dimensional problems ($d > 50$) or large batch evaluations ($N > 5,000$), exact GP Cholesky factorisation ($\mathcal{O}(N^3)$) and manual coordinate inspection would need to be replaced by sparse variational GPs, latent-space BO, or automated Trust Region Bayesian Optimization (**TuRBO**).

### 4. What risks or pitfalls should a future user be aware of when analysing these functions?
Future users must beware of four silent GP failure modes documented in this repository:
1.  **Stationarity Blindness on Sparse Spikes ($F_1$):** A global GP fitted to mostly zero data will over-smooth narrow peaks unless zero-plane pruning is applied.
2.  **ARD Length-Scale Collapse Under Noise ($F_2, F_6$):** Fitting a deterministic GP (`alpha=1e-6`) to noisy data forces length scales toward `0.01`, turning the surface into spiky noise memorisation unless a `WhiteKernel` is included.
3.  **Penalty-Cliff Variance Inflation ($F_4$):** Extreme negative outliers distort global GP covariance across the entire domain unless pruned.
4.  **Active-Learning Sampling Bias:** Because data is densely clustered around discovered peaks, the accumulated datasets must not be used to train global predictors outside the explored basins.

---

## Module 21 Datasheet Reference Summary (Motivation, Composition, Collection, Preprocessing & Maintenance)

*   **Motivation:** Iterative empirical dataset generation to characterise and maximise 8 unknown black-box functions under a 13-week sequential budget.
*   **Composition:** 8 subsets ($F_1$ to $F_8$, 2D to 8D), containing 23, 23, 28, 43, 29, 33, 43, and 53 observations respectively (**275 total evaluations**), stored in `.npy` and `.csv` formats under [`data/accumulated/`](../data/accumulated/) and raw `.txt` files under [`data/week_<1..13>_returns/`](../data/).
*   **Collection Process:** Sequential active learning (1 query/function/week across Weeks 1–13) evolving from global UCB/EI GP exploration to trust-region GP, PyTorch `64x64` ReLU NN, and analytical response-surface exploitation.
*   **Pre-processing & Uses:** Target z-score standardisation (`normalize_y=True`), active-box pruning on $F_1$ (`[0.62, 0.67]^2`), and positive-basin filtering on $F_4$ ($y > 0$). Intended strictly for sequential active-learning benchmarking and local surrogate modelling around high-performing basins.
*   **Distribution & Maintenance:** Stored and maintained in this repository ([`data/`](../data/), [`results/`](../results/), [`submissions/`](../submissions/)) under Imperial College Capstone academic guidelines.
