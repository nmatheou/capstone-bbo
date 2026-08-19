import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.svm import SVC
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import warnings
import ast

warnings.filterwarnings("ignore")

# ── Acquisition Functions ─────────────────────────────────────────────────────

def ucb(x, gp, kappa):
    x = np.atleast_2d(x)
    mean, std = gp.predict(x, return_std=True)
    return mean + kappa * std

def ei(x, gp, f_best, xi=0.1):
    x = np.atleast_2d(x)
    mean, std = gp.predict(x, return_std=True)
    with np.errstate(divide='warn'):
        imp = mean - f_best - xi
        Z = np.zeros_like(mean)
        mask = std > 0
        Z[mask] = imp[mask] / std[mask]
        ei_val = np.zeros_like(mean)
        ei_val[mask] = imp[mask] * norm.cdf(Z[mask]) + std[mask] * norm.pdf(Z[mask])
        ei_val[~mask] = 0.0
    return ei_val

def obj_func_ucb(x, gp, kappa):
    return -ucb(x, gp, kappa).item()

def obj_func_ei(x, gp, f_best, xi):
    return -ei(x, gp, f_best, xi).item()

# ── Main ──────────────────────────────────────────────────────────────────────

def run_bbo():
    output_dir  = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/antigravity"
    # FIX 1: Load from next_week_data/ (already contains original + W1 data)
    data_dir    = os.path.join(output_dir, "next_week_data")
    results_file = os.path.join(output_dir, "results_week3.txt")
    dims = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}

    # ── Parse Week 2 results ──────────────────────────────────────────────────
    results_w2_dir = os.path.join(output_dir, "results_week2")
    try:
        import re
        with open(os.path.join(results_w2_dir, "inputs.txt"), "r") as f:
            inputs_str = f.read().replace("array(", "").replace(")", "")
        inputs_str = re.sub(r'\]\s*\n\s*\[', '],\n[', inputs_str)
        new_inputs_list = ast.literal_eval(f"[{inputs_str}]")[-1]

        with open(os.path.join(results_w2_dir, "outputs.txt"), "r") as f:
            outputs_str = f.read()
        outputs_str = re.sub(r'\]\s*\n\s*\[', '],\n[', outputs_str)
        new_outputs_list = ast.literal_eval(f"[{outputs_str}]")[-1]
    except FileNotFoundError:
        print("Error: results_week2/inputs.txt or outputs.txt not found.")
        print("Please place Week 2 results in:", results_w2_dir)
        return
    except Exception as e:
        print(f"Error parsing Week 2 inputs/outputs: {e}")
        return

    # next_week_data/ will be updated in-place with the consolidated dataset
    os.makedirs(data_dir, exist_ok=True)

    # FIX 4: Global LHS seed for reproducibility
    LHS_SEED = 42

    with open(results_file, 'w') as f_out:
        for func_idx in range(1, 9):
            print(f"--- Function {func_idx} ---")
            f_out.write(f"--- Function {func_idx} ---\n")
            d = dims[func_idx]

            # ── Week 3: Kappa / Xi (decayed from Week 2, still exploratory) ──
            # Decay rule: kappa_t = kappa_{t-1} * 0.85  (from SKILL.md)
            if func_idx in [1, 2, 4, 5, 6]:
                kappa = 3.5   # 4.0 * 0.85 ≈ 3.4 → 3.5
                xi    = 0.1
            elif func_idx == 3:
                kappa = 2.0   # 2.5 * 0.85 ≈ 2.1 → 2.0
                xi    = 0.05
            elif func_idx in [7, 8]:
                kappa = 3.0   # 3.5 * 0.85 ≈ 3.0, still exploratory in high-D
                xi    = 0.1

            # ── Load accumulated data (original + W1) ─────────────────────────
            func_data_dir = os.path.join(data_dir, f"function_{func_idx}")
            X = np.load(os.path.join(func_data_dir, "initial_inputs.npy"))
            y = np.load(os.path.join(func_data_dir, "initial_outputs.npy"))

            # ── Append Week 2 result (guarded against duplicate on re-runs) ──────
            new_x = np.array(new_inputs_list[func_idx - 1])
            new_y = new_outputs_list[func_idx - 1]
            already_present = any(
                np.allclose(new_x, X[i], atol=1e-7) for i in range(len(X)))
            if not already_present:
                X = np.vstack((X, np.atleast_2d(new_x)))
                y = np.append(y, new_y)
                # Save consolidated dataset back to next_week_data/
                os.makedirs(func_data_dir, exist_ok=True)
                np.save(os.path.join(func_data_dir, "initial_inputs.npy"), X)
                np.save(os.path.join(func_data_dir, "initial_outputs.npy"), y)
            else:
                print(f"  [skip] W2 point already in dataset, not re-appending")

            print(f"  Dataset: {X.shape[0]} points | best y = {np.max(y):.6f}")

            # ── Trust Region Bounds (Functions 7 & 8) ────────────────────────
            # FIX 6: Wider initial TR (0.3 vs 0.2) to avoid over-restriction in 8D
            base_bounds = [(1e-6, 0.999999) for _ in range(d)]
            if func_idx in [7, 8]:
                best_idx = np.argmax(y)
                x_best   = X[best_idx]
                # Adaptive width: start at 0.3, but ensure it's at least as wide
                # as the spread of the top-3 observed points in each dimension
                top3_idx  = np.argsort(y)[-3:]
                spread    = np.max(X[top3_idx], axis=0) - np.min(X[top3_idx], axis=0)
                tr_width  = np.maximum(0.3, spread * 1.5)
                base_bounds = [
                    (max(1e-6,    x_best[i] - tr_width[i] / 2),
                     min(0.999999, x_best[i] + tr_width[i] / 2))
                    for i in range(d)
                ]

            # ── Fit Surrogate ─────────────────────────────────────────────────
            if func_idx == 1:
                # F1: cap length scale upper bound at 0.5 to prevent the GP
                # from declaring any dimension irrelevant with sparse data
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 0.5), nu=2.5))
            elif func_idx == 2:
                # F2: same length scale cap as F1 — prevent d2 being switched off
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 0.5), nu=2.5) +
                          WhiteKernel(noise_level=1.0,
                                      noise_level_bounds=(1e-3, 10.0)))
            elif func_idx == 4:
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 1e2), nu=1.5))
            else:
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 1e2), nu=2.5))

            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-5, normalize_y=True,
                n_restarts_optimizer=30)
            gp.fit(X, y)

            # ── FIX 2: SVC Active Subspace (Function 1) ───────────────────────
            # Threshold at 50th percentile: broader active region, more positive
            # examples for SVC — important when all y values are near-zero
            svc_model = None
            if func_idx == 1:
                threshold = np.percentile(y, 50)
                labels = (y > threshold).astype(int)
                if np.sum(labels) > 0 and np.sum(labels) < len(labels):
                    svc_model = SVC(kernel='rbf', probability=True)
                    svc_model.fit(X, labels)

            # ── FIX 3 & 5: LHS Sampling (single large draw, seeded) ──────────
            # FIX 3 & 5: Interior-biased LHS for Function 3 only
            # Function 5 boundary (dims 3&4 near 1.0) is a genuine signal — do not bias
            sampler = qmc.LatinHypercube(d=d, seed=LHS_SEED)
            X_sample = sampler.random(n=10000)

            if func_idx == 3:
                lhs_lower = np.array([0.05] * d)
                lhs_upper = np.array([0.95] * d)
            else:
                lhs_lower = np.array([b[0] for b in base_bounds])
                lhs_upper = np.array([b[1] for b in base_bounds])
            X_sample = qmc.scale(X_sample, lhs_lower, lhs_upper)

            # Apply SVC filter (Function 1)
            if svc_model is not None:
                preds = svc_model.predict(X_sample)
                if np.sum(preds == 1) > 0:
                    X_sample = X_sample[preds == 1]

            # ── Evaluate Acquisition ──────────────────────────────────────────
            if func_idx in [3, 5]:
                f_best   = np.max(y)
                acq_vals = ei(X_sample, gp, f_best, xi)
            else:
                acq_vals = ucb(X_sample, gp, kappa)

            # ── Multi-Start L-BFGS-B (top 50 seeds) ──────────────────────────
            n_seeds    = min(50, len(X_sample))
            top_X      = X_sample[np.argsort(acq_vals)[-n_seeds:]]
            full_bounds = [(1e-6, 0.999999) for _ in range(d)]

            best_x   = None
            best_val = -np.inf

            for x0 in top_X:
                if func_idx in [3, 5]:
                    res = minimize(obj_func_ei, x0,
                                   args=(gp, f_best, xi),
                                   bounds=full_bounds, method="L-BFGS-B")
                else:
                    res = minimize(obj_func_ucb, x0,
                                   args=(gp, kappa),
                                   bounds=base_bounds, method="L-BFGS-B")

                if -res.fun > best_val:
                    if svc_model is not None:
                        if svc_model.predict(res.x.reshape(1, -1))[0] == 0:
                            continue
                    best_val = -res.fun
                    best_x   = res.x

            if best_x is None:
                best_x = top_X[-1]

            best_x = np.clip(best_x, 1e-6, 0.999999)

            # ── Validation Metrics ────────────────────────────────────────────
            mean_val, std_val = gp.predict(best_x.reshape(1, -1), return_std=True)
            mean_val = mean_val[0]
            std_val  = std_val[0]

            if func_idx in [3, 5]:
                acq_score = ei(best_x, gp, f_best, xi)[0]
            else:
                acq_score = mean_val + kappa * std_val

            metrics_str = (f"Metrics -> GP Mean: {mean_val:.6f}, "
                           f"GP Std: {std_val:.6f}, Acq Score: {acq_score:.6f}")
            coord_str   = "-".join([f"{x:.6f}" for x in best_x])

            print(metrics_str)
            print(f"Coordinate String: {coord_str}\n")
            f_out.write(metrics_str + "\n")
            f_out.write(f"Coordinate String: {coord_str}\n\n")

            # ── Plot 2D Acquisition Slice ─────────────────────────────────────
            n_grid = 50
            x1_g   = np.linspace(0, 1, n_grid)
            x2_g   = np.linspace(0, 1, n_grid)
            X1, X2 = np.meshgrid(x1_g, x2_g)
            grid_pts = np.vstack((X1.flatten(), X2.flatten())).T

            grid_d = np.tile(best_x, (n_grid * n_grid, 1))
            grid_d[:, 0] = grid_pts[:, 0]
            if d > 1:
                grid_d[:, 1] = grid_pts[:, 1]

            if func_idx in [3, 5]:
                f_best_plot = np.max(y)
                acq_grid   = ei(grid_d, gp, f_best_plot, xi).reshape(n_grid, n_grid)
                acq_label  = 'EI Acquisition Score'
            else:
                acq_grid  = ucb(grid_d, gp, kappa).reshape(n_grid, n_grid)
                acq_label = 'UCB Acquisition Score'

            fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(X1, X2, acq_grid, levels=50, cmap='viridis')
            fig.colorbar(contour, ax=ax, label=acq_label)

            label_proj = '(Projected Dims 1&2)' if d > 2 else ''
            ax.scatter(X[:-1, 0], X[:-1, 1], c='red',  marker='x',
                       alpha=0.6, label=f'Historical Data {label_proj}')
            ax.scatter(X[-1, 0],  X[-1, 1],  c='cyan', marker='o',
                       edgecolors='black', label='New Point (W2)')
            ax.scatter(best_x[0], best_x[1], color='gold', s=200,
                       edgecolors='black', marker='*', label='Suggested (W3)')

            ax.set_title(f"Function {func_idx} — Week 3 Acq Surface (Dims 1 & 2)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.legend(loc='upper right', fontsize='small')

            plot_file = os.path.join(output_dir, f"plot_func{func_idx}_week3.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    run_bbo()
