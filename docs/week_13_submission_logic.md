# Week 13 Submission Logic & Final-Round Exploitation Rationale

## Function 1 (2D Sparse Needle) — LOCKED 2026-10-22
*   **Target Coordinate:** `0.628542-0.628542`
*   **Decision:** Lock in **Option B** (`+0.000001` micro-perturbation along the main diagonal from my Week 12 record) to test the unrounded analytical log-Gaussian summit while safeguarding against one-sided southwest extrapolation bias.
*   **Starting Point:** In Week 12, my analytical log-Gaussian derivation at `(0.628541, 0.628541)` delivered $\mathbf{y = 1.9999999865499047}$, achieving a 2.82× surge over Week 11 ($0.709490$) and landing within $\mathbf{1.345 \times 10^{-8}}$ of the theoretical benchmark ceiling of $\mathbf{2.00000000}$.

### Part 1: Diagnosis of One-Sided Surrogate Overshoot
1.  **Expanded Neural Network Pruning Window:**
    *   Because my Week 12 record (`0.628541, 0.628541`) fell outside the legacy `[0.65, 0.67] × [0.64, 0.67]` filter in `src/bbo_optimize.py`, I expanded the $F_1$ active-subspace pruning box to `[0.62, 0.67] × [0.62, 0.67]`, capturing all 7 positive summit points (#16–#22).
2.  **Why Both the Raw GP and Pruned NN Extrapolate Southwest:**
    *   With the 22-point dataset, the global Matern-2.5 GP ($\ell_1 = 0.0156, \ell_2 = 0.0144$) proposed `0.625571-0.623369` ($\mu = 2.035947, \sigma = 0.118120$) and the pruned 64-64 ReLU Neural Network proposed `0.623928-0.612798` ($\hat{y} = 2.300483$).
    *   Inspecting the spatial distribution of the 7 positive observations reveals the exact structural artifact driving both surrogates: all 6 prior positive points (#16–#21) sit strictly **northeast** of `(0.628541, 0.628541)` ($x_1 \in [0.628956, 0.663548], x_2 \in [0.648865, 0.664000]$), while the nearest southwest points (`0.600, 0.600` and `0.574, 0.616`) are over $0.032$ units away in the zero-response plain.
    *   Because the surrogates observe a steep upward gradient approaching `(0.628541, 0.628541)` from the northeast with **zero bracketing observations** immediately to the southwest, both models fit an unarrested upward slope past $y = 2.0$, overshooting the summit by $0.003 \text{ to } 0.016$ units.

### Part 2: Analytical Verification of the Exact Peak Center
Using the isotropic 2D Gaussian model with theoretical ceiling $A = 2.0$ and calibrated width $\sigma \approx 0.01316$:

$$
y(x_1, x_2) = 2.0 \exp\left(-\frac{(x_1 - x_1^\ast)^2 + (x_2 - x_2^\ast)^2}{2\sigma^2}\right)
$$

Solving for the Euclidean offset $r = \sqrt{(x_1 - x_1^\ast)^2 + (x_2 - x_2^\ast)^2}$ corresponding to my Week 12 return $y = 1.9999999865499047$:

$$
1 - \frac{y}{2.0} = 6.725 \times 10^{-9} \implies r = \sigma \sqrt{2 \times 6.725 \times 10^{-9}} \approx 0.01316 \times 1.160 \times 10^{-4} \approx \mathbf{1.53 \times 10^{-6}}
$$

Along a symmetric diagonal ($x_1^\ast = x_2^\ast$), this corresponds to a coordinate offset of $\Delta x_i = r / \sqrt{2} \approx \mathbf{1.08 \times 10^{-6}}$, placing the unrounded analytical summit at $\mathbf{0.6285414\dots}$ Stepping $+10^{-6}$ along both axes to **`0.628542-0.628542`** tests the nearest 6-decimal grid coordinate on the unrounded side of the peak while guaranteeing $y \approx 2.000000$.
*   **Predicted Score at Target:** Expected return $\mathbf{y^\ast \approx 2.000000}$ (GP Mean $\mu = 1.999944$, $\sigma = 0.001376$).

---

## Function 2 (2D Noisy Sim) — LOCKED 2026-10-22
*   **Target Coordinate:** `0.688100-0.387000`
*   **Decision:** Lock in **Option A** (1D GP Posterior Mean Peak along $d_1$ with $d_2$ anchored at $0.387000$) to maximize expected value on the proven ridge while eliminating inactive-dimension drift.
*   **Starting Point:** In Week 12, my query at `(0.689600, 0.387000)` returned $\mathbf{y = 0.721870}$, breaking my long-standing Week 3 record ($0.641577$ at $d_1 = 0.712094$) by $\mathbf{+0.080293}$ ($+12.5\%$) and validating that the true expected ridge crest lies near $d_1 \approx 0.688 \text{ to } 0.690$ rather than $0.712$.

### Method & Surrogate Analysis
1.  **Fitted 22-Point Noisy GP Kernel:**
    *   Surrogate: $\text{ConstantKernel}(1.0) \times \text{Matern}(\nu=2.5) + \text{WhiteKernel}$, fitted across all 22 observations:
        * $\displaystyle \boldsymbol{\ell} = [d_1: \mathbf{0.0726}, \; d_2: \mathbf{100.0000}], \quad \sigma_{\text{noise}}^2 = 0.0655 \;\; (\sigma_{\text{noise}} \approx 0.064 \text{ in raw units})$
    *   The ARD length scale $\ell_2 = 100.0$ confirms that the underlying objective is invariant along $d_2$, acting purely as a 1D noisy ridge along $d_1$.
2.  **Overruling Raw GP EI Drift on $d_2$:**
    *   The raw GP Expected Improvement optimizer proposed `0.686200-0.031962`. Because $\ell_2 = 100.0$, the GP posterior mean is virtually identical across all $d_2 \in [0, 1]$, so EI drifted to $d_2 = 0.031962$ solely to harvest a negligible $\sim 10^{-4}$ posterior variance differential far from my empirical data.
    *   Locking $d_2 = \mathbf{0.387000}$ preserves the exact empirical slice where both my all-time record ($0.721870$) and second-best score ($0.641577$) were achieved.
3.  **1D Ridge Posterior Mean Optimization:**
    *   Incorporating the Week 12 record ($0.721870$ at $d_1 = 0.689600$) alongside Point #19 ($0.608505$ at $d_1 = 0.678791$) and Point #12 ($0.641577$ at $d_1 = 0.712094$) shifts the GP posterior mean summit slightly left from $0.689600$ to $\mathbf{d_1^\ast = 0.688100}$ ($\mu = \mathbf{0.613874}, \sigma = 0.068809$), in exact agreement with the 1D quadratic parabola vertex ($0.6881$).
    *   By contrast, the unregularized Neural Network (`0.675102-0.406268`, $\hat{y} = 0.702081$) shifts onto the lower left shoulder where the GP posterior mean drops to $0.5818$.
*   **Predicted Score at Target:** GP Posterior Mean $\mu = \mathbf{0.613874}$ ($\sigma_{\text{post}} = 0.068809$, total predictive $\sigma_{\text{obs}} \approx 0.094$).

---

## Function 3 (3D Drug Discovery) — LOCKED 2026-10-22
*   **Target Coordinate:** `0.950000-0.400000-0.434766`
*   **Decision:** Lock in **Option B** (GP EI Needle #2 Left-Shoulder Peak with $(d_1, d_2)$ anchored at `(0.950000, 0.400000)`) to exploit the confirmed $d_3 \approx 0.4348$ local apex while avoiding both the inter-needle trough and the $\ell_1 = 2.0$ boundary flip.
*   **Starting Point:** My all-time record is $\mathbf{y = -0.006045}$ (Week 11 Point #26 at `0.950000, 0.400000, 0.438611`). In Week 12, my forward extrapolation along $d_3$ to `0.950000-0.400000-0.449000` returned $\mathbf{y = -0.031509}$, proving that the ascending slope between $d_3 = 0.4165$ and $d_3 = 0.4386$ is not globally monotonic and that $d_3 = 0.449000$ crossed past the crest onto a steep descending cliff.

### Method & Two-Needle Landscape Resolution
1.  **Discovery of the Bimodal $d_3$ Needle Structure:**
    *   Sorting all 27 historical observations by $d_3$ reveals why a naive 3-point parabola across Week 10 ($d_3 = 0.416489, y = -0.008597$), Week 11 ($d_3 = 0.438611, y = -0.006045$), and Week 12 ($d_3 = 0.449000, y = -0.031509$) is invalid:
        *   **Needle #1 ($d_3 \approx 0.4150$):** Queries at $d_3 = 0.414820$ ($y = -0.006152$) and $d_3 = 0.416489$ ($y = -0.008597$).
        *   **Inter-Needle Trough ($d_3 \in [0.420, 0.428]$):** Queries #18 ($d_3 = 0.425251, y = -0.023502$) and #21 ($d_3 = 0.420766, y = -0.028212$) prove a deep valley sits directly between the two peaks.
        *   **Needle #2 ($d_3 \approx 0.434 \text{ to } 0.439$):** Queries #26 ($d_3 = 0.438611, y = -0.006045$, all-time record) and #22 ($d_3 = 0.439711, y = -0.009080$).
    *   Because the naive 3-point parabola vertex ($d_3 = 0.4252$) lands directly inside the inter-needle trough ($\mu < -0.023$), it must be rejected.
2.  **Resolving the True Apex of Needle #2:**
    *   Because Query #22 ($d_3 = 0.439711, y = -0.009080$) already drops sharply just $+0.0011$ to the right of Week 11 ($d_3 = 0.438611, y = -0.006045$), Week 11 sits on the **right shoulder** of Needle #2, placing the true local summit slightly to the **left** of $0.438611$.
    *   Both the 27-point GP Expected Improvement optimizer (`best_x[2] = 0.434766`) and the 1D Matern ($\ell_3 = 0.01$) profile curve along `(0.95, 0.40, d3)` converge on the left shoulder at $\mathbf{d_3 = 0.434766}$ (a tight $0.003845$ step from my Week 11 record).
3.  **Anchoring $(d_1, d_2) = (0.950000, 0.400000)$:**
    *   In the 27-point GP, $\ell_1 = 2.000$ (hitting the upper cap), causing the unconstrained L-BFGS-B optimizer to flip $d_1$ from $0.950000$ to the opposite boundary $0.050000$ (`0.050000-0.400000-0.434766`) solely for unverified boundary uncertainty. Locking $(d_1, d_2) = (0.950000, 0.400000)$ preserves the proven high-performing pocket while targeting $d_3 = 0.434766$.
*   **Predicted Score at Target:** GP Posterior Mean $\mu = \mathbf{-0.010878}$ ($\sigma = 0.029771$), targeting a new record  > \mathbf{-0.006045}$ at the apex of Needle #2.

---

## Function 4 (4D Warehouse Logistics) — LOCKED 2026-10-22
*   **Target Coordinate:** `0.405930-0.406376-0.352329-0.422444`
*   **Decision:** Lock in **Option A1** (Positive-Basin Local GP $\nu=1.5$ Posterior Mean Peak) to eliminate global variance inflation and Kriging nugget shrinkage around my Week 12 record.
*   **Starting Point:** In Week 12, my trust-region GP query at `(0.407968, 0.405671, 0.359691, 0.420080)` achieved a **new all-time record of** $y = 0.679981$, beating my Week 9 record ($0.639193$) by $+0.040788$.

### Part 1: Why All Global GPs Under-Predict the Observed Record ($0.6644 < 0.679981$)
When fitted across all 42 historical observations, every global GP specification predicted a maximum posterior mean strictly lower than my actual Week 12 observation ($\mu \approx 0.6644 < 0.679981$). Diagnosing the GP covariance matrix revealed two compounding global distortions:
1.  **Global Variance Inflation ($114\times$):** Across all 42 points, outputs span from $-32.626$ at the outer domain cliffs to $+0.680$ at the summit ($\bar{y}_{\text{global}} = -12.66, \sigma_{y,\text{global}} = 9.74$). Within the 9 positive summit points ($y > 0$), the standard deviation is only $\sigma_{y,\text{local}} = 0.085$. When `normalize_y=True` scales by $\sigma_{y,\text{global}} = 9.74$, the effective prior pulls every prediction downward toward $-12.66$, and ARD length scales inflate to $\boldsymbol{\ell} = [2.23, 1.75, 1.98, 1.32]$ to bridge the outer cliffs.
2.  **Kriging Nugget Smoothing Across a Tight Cluster:** In the summit region, 5 historical queries sit within a Euclidean radius of $0.008 \text{ to } 0.016$ units with scores ranging from $0.5997$ to $0.6800$. With $\boldsymbol{\ell} \approx 1.3 \text{ to } 2.2$, the kernel correlation between these neighbors exceeds $0.9999$, forcing the global GP to treat the $+0.080$ local summit rise as high-frequency deviation and shrink the predicted mean at my exact Week 12 record down from $0.679981$ to $0.659633$.

### Part 2: Positive-Basin Local GP ($\nu = 1.5, n = 9$) Optimization
1.  **Pruning the Outer Cliffs ($y > 0$):**
    *   Restricting the training set to the 9 positive summit observations ($y > 0$) and fitting a Local Matern-1.5 GP (`length_scale_bounds=(0.01, 0.25)`, `alpha=1e-6`, `normalize_y=True`) yields a local kernel of $1.20^2 \times \text{Matern}(\nu=1.5)$ with ARD length scales:
        * $\displaystyle \boldsymbol{\ell}_{\text{local}} = [d_1: \mathbf{0.0302}, \; d_2: \mathbf{0.0296}, \; d_3: \mathbf{0.2500}, \; d_4: \mathbf{0.0439}]$
    *   This resolves local sensitivity across $(d_1, d_2, d_4)$ while treating $d_3$ ($\ell_3 = 0.250$) as a smoother ridge, and eliminates nugget shrinkage at my Week 12 coordinate ($\mu(\mathbf{x}_{42}) = \mathbf{0.679981}, \sigma = 0.000085$).
2.  **Why $\nu = 1.5$ (Option A1) Is Preferred Over $\nu = 2.5$ (Option A2) and the NN:**
    *   Under a twice-differentiable Matern-2.5 Local GP (Option A2), the smoother curvature assumption stretches the local length scales, causing its optimizer to slide farther away from the Week 12 cluster toward `0.402496-0.402546-0.363811-0.431731` ($\|\Delta \mathbf{x}\|_2 = 0.0139$, matching the 64-64 ReLU NN's global hyperplane tilt).
    *   By contrast, the Matern-1.5 Local GP (**Option A1**) locates an interior peak via Nelder-Mead optimization at **`0.405930-0.406376-0.352329-0.422444`**, within a very tight Euclidean step of $\|\Delta \mathbf{x}\|_2 = \mathbf{0.008027}$ from my Week 12 record, achieving a local posterior mean of $\mu_{\text{local}} = \mathbf{0.686766}$ ($\sigma_{\text{local}} = 0.026983$).
    *   *Calibration Note:* Because the Local GP is fitted on only $n = 9$ positive points, its leave-one-out cross-validation error is relatively high ($\text{LOO RMSE} = 0.1973$), so its raw normal-tail probability ($59.9\%$) should be treated as an optimistic surrogate indicator rather than a calibrated probability; crucially, the small Euclidean step ($0.0080$) keeps actual downside risk tightly bounded around the proven summit.
*   **Predicted Score at Target:** Local GP Posterior Mean $\mu_{\text{local}} = \mathbf{0.686766}$ ($\sigma_{\text{local}} = 0.026983$).

---

## Function 5 (4D Chemical Yield) — LOCKED 2026-10-22
*   **Target Coordinate:** `1.000000-1.000000-1.000000-1.000000`
*   **Decision:** Re-submit the exact mathematical hyper-corner summit `(1.000000, 1.000000, 1.000000, 1.000000)`.
*   **Starting Point:** In Week 12, eliminating the `0.999999` boundary clamp and querying `1.000000-1.000000-1.000000-1.000000` returned $\mathbf{y = 8662.482500}$, setting a **new all-time record** ($+0.0775$ above Week 8's $8662.4050$).
*   **Surrogate Consensus:** Both the 28-point Matern-2.5 GP ($\boldsymbol{\ell} = [1.91, 1.83, 1.87, 3.03]$, $\mu = 8662.415054, \sigma = 5.449470$) and the 64-64 ReLU Neural Network ($\hat{y} = 8932.059570$) converge unanimously onto `1.000000-1.000000-1.000000-1.000000`. The landscape is strictly monotonically increasing toward $x_i = 1.0$ across all four dimensions.
*   **Predicted Score at Target:** Confirmed deterministic maximum $\mathbf{y^\ast = 8662.482500}$.

---

## Function 6 (5D Cake Composite) — LOCKED 2026-10-22
*   **Target Coordinate:** `0.458352-0.330000-0.648537-0.780000-0.060000`
*   **Decision:** Lock in **Option A** (Noisy GP `Matern 2.5 + WhiteKernel` Trust-Region Box-Corner Ascent)—the only candidate across all tested models whose expected return strictly exceeds my all-time record of $-0.090586$.
*   **Starting Point:** My all-time record is $\mathbf{y = -0.090586}$ (Week 10 Point #30 at `0.450000, 0.300000, 0.662500, 0.750000, 0.087500`). In Week 12, querying `0.450000-0.300000-0.661288-0.750000-0.088712` (a Euclidean move of just $0.0017$ from Week 10) returned $\mathbf{y = -0.173497}$.

### Method & Noise-Regularized Surrogate Resolution
1.  **Diagnosing Observation Noise & Length-Scale Collapse in the Noise-Free GP:**
    *   Because Week 10 (`0.662500, 0.087500` $\to -0.090586$) and Week 12 (`0.661288, 0.088712` $\to -0.173497$) are separated by only $0.0012$ units on $d_3$ and $d_5$ yet differ by $0.0829$ in output, Function 6 carries substantial intrinsic observation noise ($\sigma_{\text{noise}} \approx 0.032 \text{ to } 0.045$).
    *   In the baseline noise-free GP (`alpha=1e-5`, no `WhiteKernel`), forcing exact interpolation across two points $0.0017$ apart with an $0.0829$ score gap caused $\ell_5$ to collapse to $\mathbf{0.0127}$ (a $45\times$ drop from Week 11's $0.573$), generating wild oscillatory artifacts that hallucinated an unphysical $+0.176511$ spike at `0.451586-0.299945-0.678538-0.744596-0.082920`.
2.  **Restoring Physical Smoothness with `WhiteKernel` & Trust-Box Ascent (Option A):**
    *   Adding a learnable `WhiteKernel` (`ConstantKernel(1.0) * Matern(nu=2.5, length_scale_bounds=(0.05, 5.0)) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 1.0))`) absorbs the observation variance ($\sigma_{\text{noise}} \approx 0.0323$ in raw units) and restores physically smooth ARD length scales across all 5 dimensions:
        * $\displaystyle \boldsymbol{\ell}_{\text{noisy}} = [d_1: 0.841, \; d_2: 0.765, \; d_3: 0.671, \; d_4: 1.781, \; d_5: 3.411]$
    *   Maximizing the Noisy GP posterior mean via `L-BFGS-B` inside a $\pm 0.03$ trust box around my Week 10 record (`[(0.42, 0.48), (0.27, 0.33), (0.63, 0.68), (0.72, 0.78), (0.06, 0.12)]`) unlocks the multi-dimensional gradient along $(d_1, d_2, d_4, d_5)$ that was held frozen during Weeks 6–12, arriving at **`0.458352-0.330000-0.648537-0.780000-0.060000`** ($\|\Delta \mathbf{x}\|_2 = 0.0531$ from Week 10).
    *   *Boundary Geometry Note:* Inside this trust box, $d_1 = 0.458352$ and $d_3 = 0.648537$ settle at interior optima, whereas $d_2 = 0.330000$, $d_4 = 0.780000$, and $d_5 = 0.060000$ pin against the $\pm 0.03$ trust-box boundaries along the GP's positive gradient slope. The trust box intentionally caps this multi-dimensional step at $\pm 0.03$ per axis to harvest the ascending slope without over-extrapolating.
3.  **Dominance Over Alternative Candidates:**
    *   Because the Noisy GP estimates that the Week 10 record ($-0.090586$) benefited from a favorable noise draw alongside nearby lower returns (Week 11: $-0.138856$, Week 12: $-0.173497$), re-querying the exact Week 10 coordinate yields a posterior mean of only $\mu_{\text{noisy}} = -0.147020$ ($\sigma = 0.035221$).
    *   By contrast, climbing the multi-dimensional slope to **Option A** achieves a posterior mean of $\mu_{\text{noisy}} = \mathbf{-0.081873}$ ($\sigma_{\text{post}} = 0.043261$; $P(y_{\text{obs}} > -0.090586) = \mathbf{56.4\%}$ including observation noise $\sigma_{\text{noise}} \approx 0.0323$), making it the **only** candidate whose expected value surpasses my all-time record ($-0.090586$).
*   **Predicted Score at Target:** Noisy GP Posterior Mean $\mu = \mathbf{-0.081873}$ ($\sigma_{\text{post}} = 0.043261$).

---

## Function 7 (6D ML Hyperparameters) — LOCKED 2026-10-22
*   **Target Coordinate:** `0.000000-0.156068-0.305066-0.315590-0.335798-0.766358`
*   **Decision:** Lock in **Option A** (Primary GP $\nu=2.5$ Trust-Region Posterior Mean Peak) to preserve the Week 12 directional gains on $(d_2, d_4, d_5, d_6)$ while restoring $d_3$ to its verified crest at $0.305066$.
*   **Starting Point:** My all-time record is $\mathbf{y = 2.201316}$ (Week 11 Point #41 at `0.000001, 0.176724, 0.283122, 0.306983, 0.353130, 0.792087`). In Week 12, my query at `0.000000-0.149707-0.154230-0.316634-0.332356-0.774035` returned $\mathbf{y = 2.137196}$ (my second-highest score in history, $-0.0641$ below Week 11).

### Method & $d_3$ Overshoot Correction
1.  **ARD Length-Scale Structure & Diagnosis of the Week 12 Dip:**
    *   Between Week 11 and Week 12, five of the six coordinates moved by less than $0.027$ units, whereas $d_3$ took a large $-0.1289$ slide from $0.283122$ down to $0.154230$ (guided by the 41-point Matern-1.5 diagnostic fit where $\ell_3 = 1.2699$).
    *   In the 42-point primary Matern-2.5 GP ($0.637^2 \times \text{Matern}(\nu=2.5)$), the fitted ARD length scales are:
        * $\displaystyle \boldsymbol{\ell} = [d_1: \mathbf{0.792}, \; d_2: \mathbf{0.588}, \; d_3: \mathbf{0.719}, \; d_4: \mathbf{0.346}, \; d_5: \mathbf{0.241}, \; d_6: \mathbf{0.296}]$
    *   Incorporating the Week 12 observation ($2.137196$ at $d_3 = 0.154230$) confirms that $d_3$ ($\ell_3 = 0.719$) has meaningful quadratic curvature peaking near $d_3 \approx 0.305$, proving that sliding $d_3$ down to $0.154230$ overshot the ridge crest.
2.  **Synthesizing Week 11's $d_3$ Crest with Week 12's $(d_2, d_4, d_5, d_6)$ Gains:**
    *   Crucially, the 42-point GP confirms that the Week 12 adjustments on $(d_2, d_4, d_5, d_6)$ were **beneficial**: combining the Week 12 coordinates on $(d_2, d_4, d_5, d_6)$ with the restored $d_3 \approx 0.305$ ridge crest increases the GP posterior mean from $2.201311$ (at Week 11) to $\mathbf{2.241241}$ ($+0.0399$ above my all-time record).
    *   Maximizing the primary GP ($\nu=2.5$) posterior mean via `L-BFGS-B` inside the trust region `[(0.0, 0.20), (0.10, 0.35), (0.15, 0.45), (0.20, 0.45), (0.25, 0.42), (0.65, 0.85)]` yields an interior peak (with only $d_1 = 0.000000$ anchored at its physical boundary) at **`0.000000-0.156068-0.305066-0.315590-0.335798-0.766358`** ($\mu = \mathbf{2.241241}, \sigma = 0.043398$, noiseless GP posterior $P(y > 2.201316) = \mathbf{82.1\%}$).
    *   By contrast, raw GP Expected Improvement (`0.000000-0.150568-0.326174-0.313526-0.330278-0.755470`, $\mu = 2.234728$) sacrifices $+0.0065$ of expected return to chase outer-flank variance on $d_3$, while the unconstrained Neural Network (`0.000000-0.000000-0.000000-0.000000-0.000000-1.000000`) extrapolates into a severe boundary penalty cliff.
*   **Predicted Score at Target:** GP Posterior Mean $\mu = \mathbf{2.241241}$ ($\sigma = 0.043398$).

---

## Function 8 (8D ML Hyperparameters) — LOCKED 2026-10-22
*   **Target Coordinate:** `0.157119-0.175371-0.120105-0.090651-0.893686-0.492256-0.191545-0.504012`
*   **Decision:** Lock in **Option C** (64-64 ReLU Neural Network Surrogate Summit) to preserve and refine the directional momentum that broke the 4-week $9.961$ plateau in Week 12.
*   **Starting Point:** In Week 12, my RSM active-subspace shift (`0.145000-0.180000-0.132000-0.098000-0.885000-0.505000-0.192000-0.500000`) achieved a **new all-time record of** $y = 9.987569$, breaking the Weeks 9–11 plateau ($9.961573$) by $\mathbf{+0.025996}$ and climbing to within $0.0124$ of the $10.0000$ ceiling.

### Part 1: Directional Momentum Analysis ($\text{W10} \to \text{W12}$ vs. $\text{W12} \to \text{W13}$)
Comparing the coordinate delta that produced the $+0.0260$ breakthrough ($\Delta(\text{W12} - \text{W10})$) against the proposed Week 13 shifts ($\Delta(\text{Option A} - \text{W12})$ for the Matern-2.5 GP vs. $\Delta(\text{Option C} - \text{W12})$ for the 64-64 ReLU NN) exposes a fundamental structural contrast:

| Dim | W10 (`9.9616`) | W12 (`9.9876`) | $\Delta(\text{W12}-\text{W10})$ | **Option C (NN)** | $\Delta(\text{Opt C}-\text{W12})$ | Option A (GP $\nu=2.5$) | $\Delta(\text{Opt A}-\text{W12})$ | Directional Verdict |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $\mathbf{d_1}$ | `0.113203` | `0.145000` | **`+0.031797`** | **`0.157119`** | **`+0.012119`** | `0.095479` | `-0.049521` | **Opt C continues** $+d_1$; Opt A reverses past W10 |
| $\mathbf{d_2}$ | `0.147122` | `0.180000` | **`+0.032878`** | **`0.175371`** | **`-0.004629`** | `0.165653` | `-0.014347` | **Opt C holds** $\approx 0.175$; Opt A pulls back $3\times$ harder |
| $\mathbf{d_3}$ | `0.115630` | `0.132000` | **`+0.016370`** | **`0.120105`** | **`-0.011895`** | `0.105801` | `-0.026199` | **Opt C settles at** $0.120$; Opt A reverses past W10 |
| $\mathbf{d_4}$ | `0.128826` | `0.098000` | **`-0.030826`** | **`0.090651`** | **`-0.007349`** | `0.106680` | `+0.008680` | **Opt C continues** $-d_4$; Opt A reverses upward |
| $\mathbf{d_5}$ | `1.000000` | `0.885000` | **`-0.115000`** | **`0.893686`** | **`+0.008686`** | `0.958744` | `+0.073744` | **Opt C stays at** $\approx 0.894$; Opt A pulls back to $1.0$ |
| $\mathbf{d_6}$ | `0.471155` | `0.505000` | **`+0.033845`** | **`0.492256`** | **`-0.012744`** | `0.464823` | `-0.040177` | **Opt C settles at** $0.492$; Opt A reverses past W10 |
| $\mathbf{d_7}$ | `0.209232` | `0.192000` | **`-0.017232`** | **`0.191545`** | **`-0.000455`** | `0.193275` | `+0.001275` | **Opt C holds** $0.1915$ |
| $\mathbf{d_8}$ | `1.000000` | `0.500000` | **`-0.500000`** | **`0.504012`** | **`+0.004012`** | `0.668989` | `+0.168989` | **Opt C holds** $0.504$; Opt A pulls back toward $1.0$ |

### Part 2: Why Option C (NN) Properly Exploits the Week 12 Breakthrough
1.  **Diagnosing GP Centroid Pullback (Option A):**
    *   Because 4 prior queries (Weeks 8–11) are tightly clustered near the Week 10 coordinate `(0.113, 0.147, 0.116, 0.129, 1.0, 0.471, 0.209, 1.0)` while Week 12 (`9.987569`) is a single isolated observation at the new interior position, the GP's broad length scales ($\boldsymbol{\ell} \in [2.8, 50.0]$) experience **sample-density gravity** toward the 4-point Week 10 cluster. Consequently, Option A reverses **all 8 directional changes** made in Week 12 (and even overshoots backward past Week 10 on $d_1, d_3, d_6$).
2.  **Coordinated Local Refinement via Option C (`0.157119-0.175371-0.120105-0.090651-0.893686-0.492256-0.191545-0.504012`):**
    *   Unlike the GP, the 64-64 ReLU Neural Network (`train_and_optimize_nn`) initiates L-BFGS gradient ascent directly from the Week 12 record (`best_idx = 51`) and follows the learned directional slope without sample-count shrinkage.
    *   Option C stays within a tight Euclidean radius of $\|\Delta \mathbf{x}\|_2 = \mathbf{0.0249}$ around Week 12 (nearly $8\times$ closer than Option A's $0.1961$), continuing the winning trajectory on $d_1$ ($+0.0121$), $d_4$ ($-0.0073$), and $d_7$ ($-0.0005$) while gently trimming $d_2, d_3,$ and $d_6$ to settle into the local quadratic bowl.
3.  **Multi-Surrogate Cross-Validation & Risk Bounds:**
    *   **64-64 ReLU Neural Network:** $\hat{y}_{\text{NN}} = \mathbf{10.005514}$ (slightly exceeding the assumed $10.0$ benchmark ceiling due to piecewise-linear ReLU extrapolation along the ascending gradient).
    *   **2nd-Order Ridge Quadratic RSM (fitted on top points $y > 9.5$):** $\hat{y}_{\text{RSM}} = \mathbf{10.000186}$.
    *   **Matern-1.5 & Matern-2.5 GPs:** Because of sample-density pullback toward the Week 10 cluster, the global Matern-2.5 GP predicts a tiny $-0.0020$ dip at Option C ($\mu_{2.5} = \mathbf{9.985399}, \sigma = 0.004806$) compared to the Week 12 record ($\mu_{2.5} = 9.987360, \sigma = 0.003794$), while the Matern-1.5 GP predicts $\mu_{1.5} = \mathbf{9.986217}$ ($\sigma = 0.0091$). Even under the conservative GP models, downside risk is bounded to $\approx 0.002$, while the RSM and NN models project full convergence toward the $10.0$ ceiling.
*   **Predicted Score at Target:** Expected return $\mathbf{y^\ast \in [9.985, 10.000]}$.

---

## Final Week 13 Submission Summary Table (All 8 Functions Locked)

| Function | Dim | Locked Week 13 Coordinate | Selected Strategy | Current Best ($y^\ast$) | Predicted Return |
| :---: | :---: | :--- | :--- | :---: | :---: |
| **F1** | 2D | `0.628542-0.628542` | **Option B:** Analytical log-Gaussian unrounded peak ($+10^{-6}$ step) | $1.99999999$ (W12) | 2.000000 ($\mu = 1.999944$) |
| **F2** | 2D | `0.688100-0.387000` | **Option A:** 1D GP posterior mean peak along $d_1$ ($d_2 = 0.387$ locked) | $0.721870$ (W12) | 0.613874 ($\mu$, $\sigma = 0.0688$) |
| **F3** | 3D | `0.950000-0.400000-0.434766` | **Option B:** GP EI Needle #2 left-shoulder apex ($(d_1, d_2)$ locked) | $-0.006045$ (W11) | -0.010878 ($\mu$, $\sigma = 0.0298$) |
| **F4** | 4D | `0.405930-0.406376-0.352329-0.422444` | **Option A1:** Positive-basin Local GP ($\nu=1.5, n=9$) mean peak | $0.679981$ (W12) | 0.686766 ($\mu_{\text{loc}}$, $\sigma = 0.0270$) |
| **F5** | 4D | `1.000000-1.000000-1.000000-1.000000` | **Unanimous GP + NN:** Exact mathematical hyper-corner summit | $8662.482500$ (W12) | 8662.482500 ($\mu = 8662.42$) |
| **F6** | 5D | `0.458352-0.330000-0.648537-0.780000-0.060000` | **Option A:** Noisy GP (`Matern 2.5 + WhiteKernel`) trust-box corner ascent | $-0.090586$ (W10) | -0.081873 ($\mu_{\text{noisy}}$, $\sigma = 0.0433$) |
| **F7** | 6D | `0.000000-0.156068-0.305066-0.315590-0.335798-0.766358` | **Option A:** Primary GP ($\nu=2.5$) trust-region mean peak ($d_3$ restored) | $2.201316$ (W11) | 2.241241 ($\mu$, $\sigma = 0.0434$) |
| **F8** | 8D | `0.157119-0.175371-0.120105-0.090651-0.893686-0.492256-0.191545-0.504012` | **Option C:** 64-64 ReLU NN local ascent peak ($\text{W10}\to\text{W12}$ continuation) | $9.987569$ (W12) | 10.0002 (y_RSM) / 9.9854 (μ_GP) |
