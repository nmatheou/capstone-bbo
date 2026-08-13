"""
bbo_optimize.py
===============
Parameterised BBO optimization script. Replaces bbo_week2.py, bbo_week3.py etc.
All paths and labels are derived from the --week argument; kappa is held fixed
until the user decides to introduce a decay schedule.

Usage
-----
    py bbo_optimize.py --week 4    # generates Week 4 coordinates
    py bbo_optimize.py --week 5    # generates Week 5 coordinates

What changes per week
---------------------
  - Reads oracle returns from  results_week{N-1}/inputs.txt + outputs.txt
  - Writes suggestions to      results_week{N}.txt
  - Saves plots to             plot_func{i}_week{N}.png
  - Appends new data point to  next_week_data/ (dedup-guarded)

What stays fixed
----------------
  - kappa / xi (no decay until explicitly changed)
  - Kernel choices per function
  - Interior LHS bias (F3 only)
  - Adaptive trust region (F7, F8)
  - SVC active subspace (F1, 50th-percentile threshold)
  - LHS seed = 42
"""

import argparse
import numpy as np
import os
import re
import ast
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.svm import SVC
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import warnings

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
        imp    = mean - f_best - xi
        Z      = np.zeros_like(mean)
        mask   = std > 0
        Z[mask]      = imp[mask] / std[mask]
        ei_val         = np.zeros_like(mean)
        ei_val[mask]   = imp[mask] * norm.cdf(Z[mask]) + std[mask] * norm.pdf(Z[mask])
        ei_val[~mask]  = 0.0
    return ei_val

def obj_func_ucb(x, gp, kappa):
    return -ucb(x, gp, kappa).item()

def obj_func_ei(x, gp, f_best, xi):
    return -ei(x, gp, f_best, xi).item()

# ── Main ──────────────────────────────────────────────────────────────────────

def run_bbo(week: int):
    output_dir = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/antigravity"
    data_dir   = os.path.join(output_dir, "next_week_data")
    dims       = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}

    # ── Derive all week-specific paths from N ─────────────────────────────────
    prev_week        = week - 1
    prev_results_dir = os.path.join(output_dir, f"results_week{prev_week}")
    results_file     = os.path.join(output_dir, f"results_week{week}.txt")

    print(f"\n{'='*60}")
    print(f"  BBO Optimize — generating Week {week} coordinates")
    print(f"  Reading oracle returns from : results_week{prev_week}/")
    print(f"  Writing suggestions to      : results_week{week}.txt")
    print(f"{'='*60}\n")

    # ── Parse previous week's oracle returns ──────────────────────────────────
    try:
        with open(os.path.join(prev_results_dir, "inputs.txt"), "r") as f:
            inputs_str = f.read().replace("array(", "").replace(")", "")
        inputs_str = re.sub(r'\]\s*\n\s*\[', '],\n[', inputs_str)
        new_inputs_list = ast.literal_eval(f"[{inputs_str}]")[-1]

        with open(os.path.join(prev_results_dir, "outputs.txt"), "r") as f:
            outputs_str = f.read()
        outputs_str = re.sub(r'\]\s*\n\s*\[', '],\n[', outputs_str)
        new_outputs_list = ast.literal_eval(f"[{outputs_str}]")[-1]
    except FileNotFoundError:
        print(f"ERROR: results_week{prev_week}/inputs.txt or outputs.txt not found.")
        print(f"Please place Week {prev_week} oracle returns in: {prev_results_dir}")
        return
    except Exception as e:
        print(f"ERROR parsing Week {prev_week} inputs/outputs: {e}")
        return

    os.makedirs(data_dir, exist_ok=True)

    # ── Fixed exploration parameters (no decay until explicitly changed) ──────
    # Adjust kappa/xi here when ready to begin exploitation phase
    KAPPA_GROUP_A = 3.5   # F1, F2, F4, F5, F6  — moderate-high exploration
    KAPPA_GROUP_B = 2.0   # F3                   — lower (landscape has small variance)
    KAPPA_GROUP_C = 3.0   # F7, F8               — still exploratory in high-D
    XI_DEFAULT    = 0.1
    XI_F3         = 0.05
    LHS_SEED      = 42

    with open(results_file, 'w') as f_out:
        for func_idx in range(1, 9):
            print(f"--- Function {func_idx} ---")
            f_out.write(f"--- Function {func_idx} ---\n")
            d = dims[func_idx]

            # ── Kappa / Xi ────────────────────────────────────────────────────
            if func_idx in [1, 2, 4, 5, 6]:
                kappa = KAPPA_GROUP_A
                xi    = XI_DEFAULT
            elif func_idx == 3:
                kappa = KAPPA_GROUP_B
                xi    = XI_F3
            else:  # 7, 8
                kappa = KAPPA_GROUP_C
                xi    = XI_DEFAULT

            # ── Load accumulated data from next_week_data/ ────────────────────
            func_data_dir = os.path.join(data_dir, f"function_{func_idx}")
            X = np.load(os.path.join(func_data_dir, "initial_inputs.npy"))
            y = np.load(os.path.join(func_data_dir, "initial_outputs.npy"))

            # ── Append previous week's oracle return (dedup-guarded) ──────────
            new_x = np.array(new_inputs_list[func_idx - 1])
            new_y = new_outputs_list[func_idx - 1]
            already_present = any(
                np.allclose(new_x, X[i], atol=1e-7) for i in range(len(X)))
            if not already_present:
                X = np.vstack((X, np.atleast_2d(new_x)))
                y = np.append(y, new_y)
                os.makedirs(func_data_dir, exist_ok=True)
                np.save(os.path.join(func_data_dir, "initial_inputs.npy"), X)
                np.save(os.path.join(func_data_dir, "initial_outputs.npy"), y)
            else:
                print(f"  [skip] W{prev_week} point already in dataset")

            print(f"  Dataset: {X.shape[0]} points | best y = {np.max(y):.6f}")

            # ── Trust Region Bounds (Functions 7 & 8) ─────────────────────────
            base_bounds = [(1e-6, 0.999999) for _ in range(d)]
            if func_idx in [7, 8]:
                best_idx  = np.argmax(y)
                x_best    = X[best_idx]
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
                # Cap length scale to prevent any dim being declared irrelevant
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 0.5), nu=2.5))
            elif func_idx == 2:
                # Same cap + WhiteKernel for noisy simulation
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 0.5), nu=2.5) +
                          WhiteKernel(noise_level=1.0,
                                      noise_level_bounds=(1e-3, 10.0)))
            elif func_idx == 4:
                # nu=1.5 for rougher landscape
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 1e2), nu=1.5))
            elif func_idx in [6, 7]:
                # Moderate cap to prevent dims hitting 100 (boundary pinning)
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 5.0), nu=2.5))
            elif func_idx == 8:
                # Conservative cap — active dims are 3–24, only target d8=100
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 50.0), nu=2.5))
            else:
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 1e2), nu=2.5))

            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-5, normalize_y=True,
                n_restarts_optimizer=30, random_state=42)
            gp.fit(X, y)

            # ── SVC Active Subspace (Function 1 only) ─────────────────────────
            svc_model = None
            if func_idx == 1:
                threshold = np.percentile(y, 50)
                labels = (y > threshold).astype(int)
                if np.sum(labels) > 0 and np.sum(labels) < len(labels):
                    svc_model = SVC(kernel='rbf', probability=True)
                    svc_model.fit(X, labels)

            # ── LHS Sampling (single seeded draw) ─────────────────────────────
            sampler   = qmc.LatinHypercube(d=d, seed=LHS_SEED)
            X_sample  = sampler.random(n=10000)

            # Interior-biased LHS for F3 only
            if func_idx == 3:
                lhs_lower = np.array([0.05] * d)
                lhs_upper = np.array([0.95] * d)
            else:
                lhs_lower = np.array([b[0] for b in base_bounds])
                lhs_upper = np.array([b[1] for b in base_bounds])
            X_sample = qmc.scale(X_sample, lhs_lower, lhs_upper)

            # Apply SVC filter
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

            grid_d        = np.tile(best_x, (n_grid * n_grid, 1))
            grid_d[:, 0]  = grid_pts[:, 0]
            if d > 1:
                grid_d[:, 1] = grid_pts[:, 1]

            if func_idx in [3, 5]:
                f_best_plot = np.max(y)
                acq_grid    = ei(grid_d, gp, f_best_plot, xi).reshape(n_grid, n_grid)
                acq_label   = 'EI Acquisition Score'
            else:
                acq_grid    = ucb(grid_d, gp, kappa).reshape(n_grid, n_grid)
                acq_label   = 'UCB Acquisition Score'

            fig, ax = plt.subplots(figsize=(8, 6))
            contour = ax.contourf(X1, X2, acq_grid, levels=50, cmap='viridis')
            fig.colorbar(contour, ax=ax, label=acq_label)

            label_proj = '(Projected Dims 1&2)' if d > 2 else ''
            ax.scatter(X[:-1, 0], X[:-1, 1], c='red', marker='x',
                       alpha=0.6, label=f'Historical Data {label_proj}')
            ax.scatter(X[-1, 0],  X[-1, 1],  c='cyan', marker='o',
                       edgecolors='black', label=f'New Point (W{prev_week})')
            ax.scatter(best_x[0], best_x[1], color='gold', s=200,
                       edgecolors='black', marker='*',
                       label=f'Suggested (W{week})')

            ax.set_title(f"Function {func_idx} — Week {week} Acq Surface (Dims 1 & 2)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.legend(loc='upper right', fontsize='small')

            plot_file = os.path.join(output_dir, f"plot_func{func_idx}_week{week}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="BBO Optimization — generate next week's coordinates")
    parser.add_argument(
        "--week", type=int, required=True,
        help="The week number to generate coordinates FOR (e.g. --week 4 reads W3 results)")
    args = parser.parse_args()

    if args.week < 2:
        print("ERROR: --week must be >= 2 (Week 1 used the original bbo_week1.py)")
    else:
        run_bbo(args.week)
