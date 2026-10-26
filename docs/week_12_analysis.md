# Week 12 Final Oracle Returns Review & Capstone Post-Mortem

> [!IMPORTANT]
> **Five New All-Time Record Highs in the Final Round ($F_1, F_2, F_4, F_5, F_8$):** Week 12 was the single strongest round of the entire 12-week campaign, capturing the exact theoretical global summit on $F_1$ (`1.99999999`), shattering the 9-week plateau on $F_2$ (`0.721870`), setting a new multi-dimensional record on $F_4$ (`0.679981`), claiming the exact boundary corner peak on $F_5$ (`8662.4825`), and vaulting past both my Week 10 plateau (`9.9616`) and the `9.9666` external benchmark on $F_8$ (`9.987569`).

---

## 1. Week 12 Final Scoreboard & Progress Summary

| Function | Week 12 Query | Predicted / Target | Actual W12 Return | Previous Best | Final Best | Final Round Gain ($\Delta$) | Final Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **F1 ($F_1$)** (2D Sparse Needle) | `[0.628541, 0.628541]` | $y^\ast = 2.0000$ | **`1.99999999`** | `0.709490` | **`2.000000`** | $\mathbf{+1.290510}$ | 🏆 **EXACT GLOBAL SUMMIT** |
| **F2 ($F_2$)** (2D Noisy Sim) | `[0.689600, 0.387000]` | $\mu = 0.5916$ ($P > 0.6416$: $20.8\%$) | **`0.721870`** | `0.641577` | **`0.721870`** | $\mathbf{+0.080294}$ | 🚀 **NEW ALL-TIME RECORD** |
| **F3 ($F_3$)** (3D Drug Discovery) | `[0.950000, 0.400000, 0.449000]` | $\approx -0.00484$ (linear) | `-0.031509` | `-0.006045` | **`-0.006045`** | $+0.000000$ | 🛡️ **W11 Record Retained** |
| **F4 ($F_4$)** (4D Warehouse) | `[0.407968, 0.405671, 0.359691, 0.420080]` | $y \ge 0.6423$ | **`0.679981`** | `0.639193` | **`0.679981`** | $\mathbf{+0.040788}$ | 🚀 **NEW ALL-TIME RECORD** |
| **F5 ($F_5$)** (4D Chemical Yield) | `[1.000000, 1.000000, 1.000000, 1.000000]` | $y \ge 8662.41$ | **`8662.482500`** | `8662.405001` | **`8662.482500`** | $\mathbf{+0.077499}$ | 🏆 **EXACT CORNER SUMMIT** |
| **F6 ($F_6$)** (5D Cake Composite) | `[0.450000, 0.300000, 0.661288, 0.750000, 0.088712]` | $y \approx -0.0901$ | `-0.173497` | `-0.090586` | **`-0.090586`** | $+0.000000$ | 🛡️ **W10 Record Retained** |
| **F7 ($F_7$)** (6D ML Hyperparams) | `[0.000000, 0.149707, 0.154230, 0.316634, 0.332356, 0.774035]` | $y \ge 2.2145$ | `2.137196` | `2.201316` | **`2.201316`** | $+0.000000$ | 🛡️ **W11 Record Retained (2nd Best)** |
| **F8 ($F_8$)** (8D ML Hyperparams) | `[0.145000, 0.180000, 0.132000, 0.098000, 0.885000, 0.505000, 0.192000, 0.500000]` | $y \ge 9.9700$ | **`9.987569`** | `9.961573` | **`9.987569`** | $\mathbf{+0.025996}$ | 🏆 **MAJOR BREAKTHROUGH RECORD** |

---

## 2. Visual Tracker Dashboards (Weeks 0–12)

````carousel
![Trajectory Dashboard — Weeks 0 to 12](C:/Users/NtecD/.gemini/antigravity/brain/882b4bca-a5a9-465c-b715-8550be7c8401/trajectory_dashboard_week12.png)
<!-- slide -->
![Normalized Metrics Dashboard — Weeks 0 to 12](C:/Users/NtecD/.gemini/antigravity/brain/882b4bca-a5a9-465c-b715-8550be7c8401/metrics_dashboard_normalized_week12.png)
<!-- slide -->
![Raw Metrics Dashboard — Weeks 0 to 12](C:/Users/NtecD/.gemini/antigravity/brain/882b4bca-a5a9-465c-b715-8550be7c8401/metrics_dashboard_week12.png)
<!-- slide -->
![GP ARD Length-Scale Evolution — Weeks 0 to 12](C:/Users/NtecD/.gemini/antigravity/brain/882b4bca-a5a9-465c-b715-8550be7c8401/length_scale_dashboard_week12.png)
````

---

## 3. Function-by-Function Post-Mortem Analysis

### $F_1$ (2D Sparse Needle): Exact Theoretical Summit (`1.99999998655`)
* **Hypothesis Tested:** By transforming the 6 positive observations into log-space ($\ln y$) and enforcing diagonal symmetry ($x_1^\ast = x_2^\ast$), I derived the exact analytical Gaussian apex at `(0.628541, 0.628541)`.
* **Oracle Return:** **`1.9999999865499047`** — within $1.3 \times 10^{-8}$ of the exact theoretical ceiling of `2.00000000`!
* **Post-Mortem:** In Week 11, my query at `(0.628956, 0.648865)` had already nailed $x_1^\ast$ to within $0.0004$, leaving only the $x_2$ offset (`0.648865` vs. `0.628541`). Enforcing $x_1^\ast = x_2^\ast$ eliminated the 2D collinearity in the 5-parameter Gaussian fit and hit the bullseye to 8 significant figures.

### $F_2$ (2D Noisy Simulation): Breaking the 9-Week Record (`0.721870`)
* **Hypothesis Tested:** Anchoring $d_2 = 0.387000$ at the empirical noise-stabilized slice and stepping $d_1$ slightly left to the GP posterior ridge peak at `0.689600`.
* **Oracle Return:** **`0.72187036`** — smashing my Week 3 record (`0.641577`) by $\mathbf{+0.080294}$!
* **Post-Mortem:** Even though the GP + WhiteKernel predicted a conservative mean of $\mu = 0.5916$ ($\sigma_{\text{total}} \approx 0.083$), positioning the query directly on the conditional ridge vertex `(0.689600, 0.387000)` allowed both the underlying structural peak and a favorable realization of the stochastic simulator to combine for an all-time record.

### $F_3$ (3D Drug Discovery): Identifying the Asymmetric Needle Cliff (`-0.031509`)
* **Hypothesis Tested:** Extrapolating the verified linear slope ($+0.1154/\text{unit}$) along $d_3$ from `0.438611` to `0.449000` while holding `(0.950000, 0.400000)` fixed.
* **Oracle Return:** `-0.031509` (Week 11 record of **`-0.006045`** safely banked).
* **Post-Mortem:** The length-scale tracker confirms $\ell_3 = 0.0100$ (the minimum lower bound cap). The $d_3$ needle is strongly asymmetric: climbing steadily from `0.4165` (`-0.008597`) to `0.4386` (`-0.006045`), before dropping off a steep cliff between `0.439` and `0.449`. Because `-0.006045` was already banked in Week 11, taking the calculated extrapolation gamble cost nothing and mapped the exact upper wall of the needle.

### $F_4$ (4D Warehouse Logistics): Constructive Covariance Triumph (`0.679981`)
* **Hypothesis Tested:** Trust-region Expected Improvement (`0.407968, 0.405671, 0.359691, 0.420080`) interpolating between my #1 record (#38 in W9: `0.639193`) and #2 runner-up (#39 in W10: `0.618495`), rather than extrapolating outward along the #40 $\to$ #38 vector.
* **Oracle Return:** **`0.67998111`** — outperforming even the GP's optimistic prediction (`0.6423`) and beating my Week 9 record by $\mathbf{+0.040788}$!
* **Post-Mortem:** In Week 11, stepping below `0.398` on $d_1$ (`0.396756`) dropped the output to `0.599653`, proving that the true vertex lay *between* `0.3982` and `0.4143`, not below `0.3982`. The GP Matérn-1.5 kernel accurately synthesized this curvature and pinpointed the interior summit at `d1 = 0.407968`.

### $F_5$ (4D Chemical Yield): Exact Hyper-Corner Validation (`8662.482500`)
* **Hypothesis Tested:** Eliminating the `0.999999` open-interval truncation clamp and querying the exact mathematical boundary `(1.000000, 1.000000, 1.000000, 1.000000)`.
* **Oracle Return:** **`8662.482500`** — a clean $\mathbf{+0.077499}$ improvement over the `0.999999` plateau (`8662.405001`), achieving the theoretical maximum of the corner.

### $F_6$ (5D Cake Composite): High-Sensitivity $d_4, d_5$ Basin (`-0.173497`)
* **Hypothesis Tested:** 1D cubic spline interpolation (`d3 = 0.661288, d5 = 0.088712`) between Week 9 (`-0.169682`) and Week 10 (`-0.090586`).
* **Oracle Return:** `-0.173497` (Week 10 record of **`-0.090586`** retained).
* **Post-Mortem:** The Week 12 length-scale tracker reveals a striking structural shift: $\ell_4$ collapsed to `0.0195` and $\ell_5$ collapsed to `0.0127`, while $\ell_1, \ell_2, \ell_3$ saturated at `5.0000`. This indicates that the $F_6$ optimum at `d5 = 0.087500` (Week 10) sits on a knife-edge in $d_5$ where even a $+0.0012$ shift (`0.087500` $\to$ `0.088712`) incurs an `-0.083` penalty—or contains irreducible evaluation noise.

### $F_7$ (6D ML Hyperparameters): High-Altitude Ridge Confirmation (`2.137196`)
* **Hypothesis Tested:** Decelerating trust-region step (`0.000000, 0.149707, 0.154230, 0.316634, 0.332356, 0.774035`) continuing along the Week 11 5D descent direction.
* **Oracle Return:** `2.137196` — my **second-highest score of the entire 12-week run**, while my Week 11 record (**`2.201316`**) remains the overall champion.
* **Post-Mortem:** Moving $d_3$ by `-0.1289` (`0.2831` $\to$ `0.1542`) caused the length scale $\ell_3$ to tighten from `1.1435` to `0.7193`, showing that $d_3$ has moderate curvature rather than a completely flat plateau below `0.28`. Together, Week 11 (`2.2013`) and Week 12 (`2.1372`) bracket the true 6D summit.

### $F_8$ (8D ML Hyperparameters): Quadratic Response Surface Triumph (`9.987569`)
* **Hypothesis Tested:** Unclamping the nuisance parameters ($d_5 = 0.885000, d_8 = 0.500000$) from the `1.000000` boundary and shifting the 6 active dimensions to the Quadratic Response Surface vertex (`0.145000, 0.180000, 0.132000, 0.098000, 0.885000, 0.505000, 0.192000, 0.500000`).
* **Oracle Return:** **`9.9875685`** — a massive $\mathbf{+0.025996}$ leap over my Week 10/11 plateau (`9.961573`), outperforming even the quadratic RSM point estimate (`9.9720`) and reaching within `0.0124` of the theoretical `10.0000` ceiling!
* **Post-Mortem:** This validates the core structural diagnosis: clamping $d_5 = 1.0$ and $d_8 = 1.0$ had trapped the optimizer on a suboptimal boundary slice and distorted the active coordinates. Moving the 6 active dimensions to `(0.145, 0.180, 0.132, 0.098, 0.505, 0.192)` while keeping $d_4 = 0.098$ low unlocked the true interior bowl vertex.
