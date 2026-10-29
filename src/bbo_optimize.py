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
import json
from datetime import datetime
import numpy as np
import os
import sys
import re
import ast
import matplotlib.pyplot as plt

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from paths import (
    REPO_ROOT,
    get_output_dir,
    get_config_path,
    validate_project_structure,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.svm import SVC
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import warnings
import torch
import torch.nn as nn
import torch.optim as optim


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


def train_and_optimize_nn(X, y, func_idx):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # F1 Pruning Logic
    if func_idx == 1:
        mask = (X[:, 0] >= 0.62) & (X[:, 0] <= 0.67) & (X[:, 1] >= 0.62) & (X[:, 1] <= 0.67)
        if np.sum(mask) > 0:
            X_t = torch.tensor(X[mask], dtype=torch.float32)
            y_t = torch.tensor(y[mask], dtype=torch.float32).view(-1, 1)

    y_mean = y_t.mean()
    y_std = y_t.std() + 1e-8
    y_norm = (y_t - y_mean) / y_std

    d = X_t.shape[1]
    
    class ShallowNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.net(x)
    
    torch.manual_seed(42)
    model = ShallowNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(1500):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_norm)
        loss.backward()
        optimizer.step()
        
    best_idx = torch.argmax(y_t).item()
    best_x = X_t[best_idx].clone().detach().requires_grad_(True)
    opt_max = optim.LBFGS([best_x], lr=0.01, max_iter=1000)

    def closure():
        opt_max.zero_grad()
        loss = -model(best_x)
        loss.backward()
        return loss

    opt_max.step(closure)
    with torch.no_grad():
        final_x = torch.clamp(best_x, 0.0, 1.0)
        pred_y_norm = model(final_x)
        pred_y = pred_y_norm * y_std + y_mean
        
    return final_x.numpy(), pred_y.item()


def compute_final_refinement(func_idx, X, y, gp, best_x, nn_best_x):
    """
    Computes the locked final-round (Week 13) exploitation refinement coordinate
    for each function, resolving global variance inflation, nugget shrinkage,
    and inactive-dimension drift.
    """
    x_best = X[np.argmax(y)]

    if func_idx == 1:
        # F1: Step +1e-6 along the diagonal from W12 record (0.628541, 0.628541)
        # to test the unrounded analytical log-Gaussian center (0.6285414...).
        final_x = np.round(x_best + 1e-6, 6)
        mu, std = gp.predict(final_x.reshape(1, -1), return_std=True)
        return final_x, f"Analytical Log-Gaussian Peak (+1e-6 Step) | GP Mean: {mu[0]:.6f}, Std: {std[0]:.6f}"

    elif func_idx == 2:
        # F2: Lock d2 at proven record slice (0.387000) to prevent ell_2=100 drift,
        # and maximize GP posterior mean along d1 ridge.
        d1_grid = np.linspace(0.65, 0.73, 801)
        pts = np.column_stack([d1_grid, np.full_like(d1_grid, x_best[1])])
        mu, std = gp.predict(pts, return_std=True)
        idx = np.argmax(mu)
        final_x = pts[idx]
        return final_x, f"1D Ridge Posterior Mean Peak (d2 locked at 0.387000) | GP Mean: {mu[idx]:.6f}, Std: {std[idx]:.6f}"

    elif func_idx == 3:
        # F3: Lock (d1, d2) = (0.950000, 0.400000) at proven boundary pocket and
        # use GP EI coordinate on Needle #2 left shoulder (d3 = 0.434766).
        final_x = np.array([0.95, 0.40, best_x[2]])
        mu, std = gp.predict(final_x.reshape(1, -1), return_std=True)
        return final_x, f"Boundary Pocket Lock + Needle #2 Shoulder | GP Mean: {mu[0]:.6f}, Std: {std[0]:.6f}"

    elif func_idx == 4:
        # F4: Prune outer cliffs (keep y > 0 summit basin, n=9) to eliminate global
        # variance inflation and nugget shrinkage; maximize Local GP (nu=1.5) mean.
        mask = y > 0
        k_loc = ConstantKernel(1.0) * Matern(
            length_scale=np.ones(4) * 0.05,
            length_scale_bounds=(0.01, 0.25),
            nu=1.5
        )
        gp_loc = GaussianProcessRegressor(
            kernel=k_loc, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=30, random_state=42
        )
        gp_loc.fit(X[mask], y[mask])
        res = minimize(
            lambda x: -gp_loc.predict(x.reshape(1, -1))[0],
            X[-1],
            method='Nelder-Mead',
            options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 5000}
        )
        final_x = np.clip(res.x, 0.0, 1.0)
        mu, std = gp_loc.predict(final_x.reshape(1, -1), return_std=True)
        return final_x, f"Positive-Basin Local GP (nu=1.5, n=9) Mean Peak | Local GP Mean: {mu[0]:.6f}, Std: {std[0]:.6f}"

    elif func_idx == 5:
        # F5: Unanimous GP EI + NN hyper-corner maximum.
        final_x = best_x.copy()
        mu, std = gp.predict(final_x.reshape(1, -1), return_std=True)
        return final_x, f"Unanimous GP + NN Hyper-Corner Summit | GP Mean: {mu[0]:.6f}, Std: {std[0]:.6f}"

    elif func_idx == 6:
        # F6: Fit Noisy GP (Matern 2.5 + WhiteKernel) to absorb observation noise
        # (preventing ell_5 collapse) and maximize posterior mean inside +/-0.03 trust box.
        k_noisy = (
            ConstantKernel(1.0) * Matern(
                length_scale=np.ones(5),
                length_scale_bounds=(0.05, 5.0),
                nu=2.5
            ) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 1.0))
        )
        gp_noisy = GaussianProcessRegressor(
            kernel=k_noisy, alpha=1e-5, normalize_y=True,
            n_restarts_optimizer=30, random_state=42
        )
        gp_noisy.fit(X, y)
        tr_bnds = [(0.42, 0.48), (0.27, 0.33), (0.63, 0.68), (0.72, 0.78), (0.06, 0.12)]
        res = minimize(
            lambda x: -gp_noisy.predict(x.reshape(1, -1))[0],
            x_best,
            bounds=tr_bnds,
            method='L-BFGS-B'
        )
        final_x = np.clip(res.x, 0.0, 1.0)
        mu, std = gp_noisy.predict(final_x.reshape(1, -1), return_std=True)
        return final_x, f"Noisy GP (Matern 2.5 + WhiteKernel) Trust-Box Corner Ascent | Noisy GP Mean: {mu[0]:.6f}, Std: {std[0]:.6f}"

    elif func_idx == 7:
        # F7: Maximize primary GP (nu=2.5) posterior mean inside trust region around W11/W12
        # to preserve (d2, d4, d5, d6) gains while restoring d3 to 0.305066.
        tr_bnds = [(0.0, 0.20), (0.10, 0.35), (0.15, 0.45), (0.20, 0.45), (0.25, 0.42), (0.65, 0.85)]
        res = minimize(
            lambda x: -gp.predict(x.reshape(1, -1))[0],
            x_best,
            bounds=tr_bnds,
            method='L-BFGS-B'
        )
        final_x = np.clip(res.x, 0.0, 1.0)
        mu, std = gp.predict(final_x.reshape(1, -1), return_std=True)
        return final_x, f"GP (nu=2.5) Trust-Region Posterior Mean Peak | GP Mean: {mu[0]:.6f}, Std: {std[0]:.6f}"

    elif func_idx == 8:
        # F8: Select 64-64 ReLU Neural Network surrogate gradient-ascent peak (Option C),
        # which continues the successful W10->W12 directional momentum around W12 record.
        final_x = nn_best_x.copy()
        mu, std = gp.predict(final_x.reshape(1, -1), return_std=True)
        return final_x, f"64-64 ReLU NN Surrogate Peak (Option C) | GP Mean: {mu[0]:.6f}, Std: {std[0]:.6f}"

def run_bbo(week: int, dry_run: bool = False, output_dir: str | None = None):
    output_dir = get_output_dir(cli_arg=output_dir, prompt_if_first_run=True)
    warnings = validate_project_structure(output_dir, week=week)
    for w in warnings:
        print(f"  [Notice] {w}")
    data_dir   = os.path.join(output_dir, "data", "accumulated")
    dims       = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}

    # -- Derive all week-specific paths from N ---------------------------------
    prev_week        = week - 1
    prev_results_dir = os.path.join(output_dir, "data", f"week_{prev_week}_returns")
    results_dir      = os.path.join(output_dir, "results", f"week_{week}")
    results_file     = os.path.join(results_dir, "suggestions.txt")
    logs_dir         = os.path.join(output_dir, "logs")
    log_file         = os.path.join(logs_dir, "run_logs.txt")

    if not dry_run:
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BBO Optimize - generating Week {week} coordinates")
    print(f"  Reading oracle returns from : data/week_{prev_week}_returns/")
    if dry_run:
        print("  Dry-run: no writes to accumulated data, results, logs, or submissions")
    else:
        print(f"  Writing suggestions to      : results/week_{week}/suggestions.txt")
    print(f"{'='*60}\n")

    # -- Parse previous week's oracle returns ----------------------------------
    try:
        with open(os.path.join(prev_results_dir, "inputs.txt"), "r") as f:
            inputs_str = f.read().replace("array(", "").replace(")", "")
        import ast
        inputs_str = re.sub(r'\]\s*\n\s*\[', '],\n[', inputs_str)
        new_inputs_list = ast.literal_eval(f"[{inputs_str}]")[-1]

        with open(os.path.join(prev_results_dir, "outputs.txt"), "r") as f:
            outputs_str = f.read()
        outputs_str = re.sub(r'\]\s*\n\s*\[', '],\n[', outputs_str)
        new_outputs_list = ast.literal_eval(f"[{outputs_str}]")[-1]
    except FileNotFoundError:
        print(f"ERROR: data/week_{prev_week}_returns/inputs.txt or outputs.txt not found.")
        return
    except Exception as e:
        print(f"ERROR parsing Week {prev_week} inputs/outputs: {e}")
        return

    os.makedirs(data_dir, exist_ok=True)

    # -- Load Hyperparameters --------------------------------------------------
    config_path = get_config_path(output_dir)
    with open(config_path, "r") as f:
        config = json.load(f)
    
    KAPPA_GROUP_A = config.get("KAPPA_GROUP_A", 3.5)
    KAPPA_GROUP_B = config.get("KAPPA_GROUP_B", 2.0)
    KAPPA_GROUP_C = config.get("KAPPA_GROUP_C", 3.0)
    XI_DEFAULT    = config.get("XI_DEFAULT", 0.01)
    XI_F3         = config.get("XI_F3", 0.001)
    XI_F1         = config.get("XI_F1", 0.0001)
    LHS_SEED      = config.get("LHS_SEED", 42)

    final_submissions = []
    results_target = os.devnull if dry_run else results_file
    log_target = os.devnull if dry_run else log_file
    with open(results_target, 'w') as f_out, open(log_target, 'a') as f_log:
        f_log.write(f"\n--- Run Timestamp: {datetime.now().isoformat()} | Target Week: {week} ---\n")
        for func_idx in range(1, 9):
            print(f"--- Function {func_idx} ---")
            f_out.write(f"--- Function {func_idx} ---\n")
            d = dims[func_idx]

            # ── Kappa / Xi ────────────────────────────────────────────────────
            if func_idx == 1:
                kappa = KAPPA_GROUP_A
                xi    = XI_F1
            elif func_idx in [2, 4, 5, 6]:
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
                if dry_run:
                    print(f"  [dry-run] would append W{prev_week} point (not saved)")
                else:
                    os.makedirs(func_data_dir, exist_ok=True)
                    np.save(os.path.join(func_data_dir, "initial_inputs.npy"), X)
                    np.save(os.path.join(func_data_dir, "initial_outputs.npy"), y)
            else:
                print(f"  [skip] W{prev_week} point already in dataset")

            print(f"  Dataset: {X.shape[0]} points | best y = {np.max(y):.6f}")

            # ── Trust Region Bounds (Functions 4, 6, 7 & 8) ───────────────────
            base_bounds = [(0.0, 1.0) for _ in range(d)]
            if func_idx in [4, 6, 7, 8]:
                best_idx  = np.argmax(y)
                x_best    = X[best_idx]
                top3_idx  = np.argsort(y)[-3:]
                spread    = np.max(X[top3_idx], axis=0) - np.min(X[top3_idx], axis=0)
                tr_width  = np.maximum(0.3, spread * 1.5)
                base_bounds = [
                    (max(0.0, x_best[i] - tr_width[i] / 2),
                     min(1.0, x_best[i] + tr_width[i] / 2))
                    for i in range(d)
                ]

            # ── Fit Surrogate ─────────────────────────────────────────────────
            if func_idx == 1:
                # Cap length scale to prevent any dim being declared irrelevant
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 0.5), nu=2.5))
            elif func_idx == 2:
                # Cap removed — let GP freely learn d2 is irrelevant (large length scale)
                # WhiteKernel retained to handle noisy simulation variance
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 1e2), nu=2.5) +
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
            elif func_idx == 3:
                # Tight cap — prevent d1 from hitting 100 (exploitation phase)
                # Forces GP to model local structure around interior peak
                kernel = (ConstantKernel(1.0) *
                          Matern(length_scale=np.ones(d),
                                 length_scale_bounds=(1e-2, 2.0), nu=2.5))
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

            # Exploitation-phase LHS for F3: tighten around known best region
            if func_idx == 3:
                lhs_lower = np.array([0.05, 0.40, 0.30])
                lhs_upper = np.array([0.95, 0.65, 0.50])
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
            # All functions now use EI (exploitation phase)
            f_best   = np.max(y)
            acq_vals = ei(X_sample, gp, f_best, xi)

            # ── Multi-Start L-BFGS-B (top 50 seeds) ──────────────────────────
            n_seeds    = min(50, len(X_sample))
            top_X      = X_sample[np.argsort(acq_vals)[-n_seeds:]]
            if func_idx == 3:
                full_bounds = [(0.05, 0.95), (0.40, 0.65), (0.30, 0.50)]
            else:
                full_bounds = [(0.0, 1.0) for _ in range(d)]

            best_x   = None
            best_val = -np.inf

            for x0 in top_X:
                res = minimize(obj_func_ei, x0,
                               args=(gp, f_best, xi),
                               bounds=full_bounds, method="L-BFGS-B")

                if -res.fun > best_val:
                    if svc_model is not None:
                        if svc_model.predict(res.x.reshape(1, -1))[0] == 0:
                            continue
                    best_val = -res.fun
                    best_x   = res.x

            if best_x is None:
                best_x = top_X[-1]

            best_x = np.clip(best_x, 0.0, 1.0)

            # ── Validation Metrics ────────────────────────────────────────────
            mean_val, std_val = gp.predict(best_x.reshape(1, -1), return_std=True)
            mean_val = mean_val[0]
            std_val  = std_val[0]

            acq_score = ei(best_x, gp, f_best, xi)[0]

            metrics_str = (f"Metrics -> GP Mean: {mean_val:.6f}, "
                           f"GP Std: {std_val:.6f}, Acq Score: {acq_score:.6f}")
            coord_str   = "-".join([f"{x:.6f}" for x in best_x])

            print(metrics_str)
            print(f"Coordinate String: {coord_str}\n")
            f_out.write(metrics_str + "\n")
            f_out.write(f"Coordinate String: {coord_str}\n\n")

            # --- Neural Network Validation ---
            nn_best_x = best_x.copy()
            try:
                nn_best_x, nn_pred_y = train_and_optimize_nn(X, y, func_idx)
                nn_coord_str = "-".join([f"{x:.6f}" for x in nn_best_x])
                nn_str = f"NN Metrics -> Predicted Max Score: {nn_pred_y:.6f}\nNN Coordinate String: {nn_coord_str}"
                print(nn_str + "\n")
                f_out.write(nn_str + "\n\n")
                f_log.write(f"  NN Suggestion: {nn_coord_str} | NN Pred: {nn_pred_y:.6f}\n")
            except Exception as e:
                print(f"NN Training Failed: {e}")
                f_log.write(f"  NN Training Failed: {e}\n")

            # --- Final-Round (Week 13+) Exploitation Refinement ---
            if week >= 13:
                final_x, final_desc = compute_final_refinement(
                    func_idx, X, y, gp, best_x, nn_best_x
                )
                final_coord_str = "-".join([f"{x:.6f}" for x in final_x])
                final_str = (
                    f"Final Refinement -> {final_desc}\n"
                    f"Final Submitted Coordinate String: {final_coord_str}"
                )
                print(final_str + "\n")
                f_out.write(final_str + "\n\n")
                f_log.write(f"  Final Submitted: {final_coord_str} ({final_desc})\n")
                final_submissions.append(final_coord_str)
                best_x = final_x
            
            # --- Logging ---
            acq_name = 'EI'
            f_log.write(f"Func {func_idx}: Dataset size: {len(X)} | Best Y: {np.max(y):.6f}\n")
            f_log.write(f"  Acquisition: {acq_name} | Kappa: {kappa} | Xi: {xi}\n")
            if hasattr(gp.kernel_, 'k1') and hasattr(gp.kernel_.k1, 'k2'):
                f_log.write(f"  Length Scales: {gp.kernel_.k1.k2.length_scale}\n")
            elif hasattr(gp.kernel_, 'k2') and hasattr(gp.kernel_.k2, 'length_scale'):
                f_log.write(f"  Length Scales: {gp.kernel_.k2.length_scale}\n")
            elif hasattr(gp.kernel_, 'length_scale'):
                f_log.write(f"  Length Scales: {gp.kernel_.length_scale}\n")
            else:
                f_log.write(f"  Length Scales: [Unable to extract]\n")
            f_log.write(f"  Suggestion: {coord_str} | Acq Score: {best_val:.6f}\n\n")

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

            f_best_plot = np.max(y)
            acq_grid    = ei(grid_d, gp, f_best_plot, xi).reshape(n_grid, n_grid)
            acq_label   = 'EI Acquisition Score'

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

            if not dry_run:
                plot_dir = os.path.join(results_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_file = os.path.join(plot_dir, f"plot_func{func_idx}.png")
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()

    if week >= 13 and len(final_submissions) == 8:
        if dry_run:
            print("  [dry-run] skipping write of submissions/week_"
                  f"{week}/suggestions.txt")
        else:
            sub_dir = os.path.join(output_dir, "submissions", f"week_{week}")
            os.makedirs(sub_dir, exist_ok=True)
            sub_file = os.path.join(sub_dir, "suggestions.txt")
            with open(sub_file, "w") as f_sub:
                f_sub.write("\n".join(final_submissions) + "\n")
            print(f"  Saved locked final submissions to : submissions/week_{week}/suggestions.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="BBO Optimization — generate next week's coordinates")
    parser.add_argument(
        "--week", type=int, required=True,
        help="The week number to generate coordinates FOR (e.g. --week 4 reads W3 results)")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Base directory for project outputs (default: prompt on first run, cached in configs/runtime_config.json)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fit and print suggestions without writing accumulated .npy or locked submissions")
    args = parser.parse_args()

    if args.week < 2:
        print("ERROR: --week must be >= 2 (Week 1 used the original bbo_week1.py)")
    else:
        run_bbo(args.week, dry_run=args.dry_run, output_dir=args.output_dir)
