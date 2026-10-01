"""
dim_sensitivity_sweep.py
========================
For each function, holds all dims at the current best-known input x* and sweeps
each dimension individually across [0, 1] in 100 steps, plotting the GP mean
+/- 2*std. A flat line = the GP treats that dimension as irrelevant.

Sensitivity Score = (max_mean - min_mean) / gp_std_at_xstar
  < 0.5  -> effectively irrelevant (variation within noise floor)
  0.5-2  -> weakly relevant
  > 2    -> clearly relevant

Outputs
-------
1. Console table: sensitivity scores per function per dimension
2. dim_sensitivity_week{N}.png  -- per-function, per-dim sweep plots
"""

import sys
import re
import ast
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
plt.style.use("dark_background")

# ── Paths ─────────────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--week', type=int, required=True, help="Target week (e.g. 5)")
args, unknown = parser.parse_known_args()
TARGET_WEEK = args.week

BASE_DIR   = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/M12/Data/M12"
OUTPUT_DIR = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/antigravity"
WEEKS_DIRS = {w: os.path.join(OUTPUT_DIR, "data", f"week_{w}_returns") for w in range(1, TARGET_WEEK + 1)}

DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
FUNC_NAMES = {
    1: "F1 — 2D Sparse Needle",
    2: "F2 — 2D Noisy Sim",
    3: "F3 — 3D Drug Discovery",
    4: "F4 — 4D Warehouse",
    5: "F5 — 4D Chemical Yield",
    6: "F6 — 5D Cake Composite",
    7: "F7 — 6D ML Hyperparams",
    8: "F8 — 8D ML Hyperparams",
}
SWEEP_STEPS = 100
N_SWEEP = np.linspace(0.0, 1.0, SWEEP_STEPS)

DIM_COLOURS = [
    "#00E5FF", "#FFD54F", "#A5D6A7", "#FF7043",
    "#CE93D8", "#80DEEA", "#FFCC80", "#F48FB1"
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all_data(fi):
    """Return accumulated (X, y) across all available weeks."""
    f_dir = os.path.join(BASE_DIR, f"function_{fi}")
    X = np.load(os.path.join(f_dir, "initial_inputs.npy"))
    y = np.load(os.path.join(f_dir, "initial_outputs.npy"))

    for week, week_dir in WEEKS_DIRS.items():
        try:
            with open(os.path.join(week_dir, "inputs.txt"), "r") as f:
                raw = f.read().replace("array(", "").replace(")", "")
            raw = re.sub(r'\]\s*\n\s*\[', '],\n[', raw)
            inputs_list = ast.literal_eval(f"[{raw}]")[-1]

            with open(os.path.join(week_dir, "outputs.txt"), "r") as f:
                out_raw = f.read()
            out_raw = re.sub(r'\]\s*\n\s*\[', '],\n[', out_raw)
            outputs_list = ast.literal_eval(f"[{out_raw}]")[-1]

            X = np.vstack([X, np.atleast_2d(inputs_list[fi - 1])])
            y = np.append(y, outputs_list[fi - 1])
        except FileNotFoundError:
            break

    return X, y


def build_kernel(func_idx, d):
    if func_idx == 1:
        return (ConstantKernel(1.0) *
                Matern(length_scale=np.ones(d),
                       length_scale_bounds=(1e-2, 0.5), nu=2.5))
    elif func_idx == 2:
        return (ConstantKernel(1.0) *
                Matern(length_scale=np.ones(d),
                       length_scale_bounds=(1e-2, 0.5), nu=2.5) +
                WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 10.0)))
    elif func_idx == 4:
        return (ConstantKernel(1.0) *
                Matern(length_scale=np.ones(d),
                       length_scale_bounds=(1e-2, 1e2), nu=1.5))
    elif func_idx in [6, 7]:
        # Moderate cap to prevent dims hitting 100 (boundary pinning)
        return (ConstantKernel(1.0) *
                Matern(length_scale=np.ones(d),
                       length_scale_bounds=(1e-2, 5.0), nu=2.5))
    elif func_idx == 8:
        # Conservative cap — active dims are 3–24, only target d8=100
        return (ConstantKernel(1.0) *
                Matern(length_scale=np.ones(d),
                       length_scale_bounds=(1e-2, 50.0), nu=2.5))
    else:
        return (ConstantKernel(1.0) *
                Matern(length_scale=np.ones(d),
                       length_scale_bounds=(1e-2, 1e2), nu=2.5))


def sensitivity_sweep(gp, x_star, dim_idx, d, steps=SWEEP_STEPS):
    """Sweep dim_idx from 0 to 1, holding all other dims at x_star values."""
    grid = np.tile(x_star, (steps, 1))
    grid[:, dim_idx] = N_SWEEP
    means, stds = gp.predict(grid, return_std=True)
    return means.flatten(), stds.flatten()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Determine how many completed weeks we have for output naming
    latest_week = 0
    for w in sorted(WEEKS_DIRS.keys()):
        if os.path.exists(os.path.join(WEEKS_DIRS[w], "outputs.txt")):
            latest_week = w

    tracker_dir = os.path.join(OUTPUT_DIR,
                               f"results/week_{latest_week}/progress")
    os.makedirs(tracker_dir, exist_ok=True)

    print("\nFitting GPs and running sensitivity sweeps...")

    # One figure per function, dims as subplots within
    all_scores = {}   # {fi: {dim_i: score}}

    # Layout: 2 rows x 4 cols of function panels; within each panel, dims stacked
    # We'll create one big figure with a custom grid

    fig = plt.figure(figsize=(24, 14))
    fig.suptitle(
        f"GP Dimension Sensitivity Sweep (Week {latest_week} data)\n"
        "Flat line within shaded band = dimension effectively irrelevant to GP",
        fontsize=13, fontweight="bold", color="white", y=1.01)
    fig.patch.set_facecolor("#0d0d0d")

    outer = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    print(f"\n{'Function':<24} {'Dim':<6} {'Sensitivity Score':>20}  {'Assessment':>14}")
    print("=" * 72)

    for fi in range(1, 9):
        d = DIMS[fi]
        X, y = load_all_data(fi)
        x_star = X[np.argmax(y)]   # best known input

        kernel = build_kernel(fi, d)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-5,
            normalize_y=True, n_restarts_optimizer=10, random_state=42)
        gp.fit(X, y)

        # GP std at x_star (our uncertainty baseline)
        _, std_star = gp.predict(x_star.reshape(1, -1), return_std=True)
        std_baseline = float(std_star[0]) + 1e-9

        all_scores[fi] = {}

        row = (fi - 1) // 4
        col = (fi - 1) % 4
        inner = gridspec.GridSpecFromSubplotSpec(
            d, 1, subplot_spec=outer[row, col], hspace=0.05)

        for dim_i in range(d):
            ax = fig.add_subplot(inner[dim_i])
            ax.set_facecolor("#111111")

            means, stds = sensitivity_sweep(gp, x_star, dim_i, d)

            # Sensitivity score
            score = (np.max(means) - np.min(means)) / std_baseline
            all_scores[fi][dim_i] = score

            # Classification
            if score < 0.5:
                label = "IRRELEVANT?"
                colour = "#888888"
            elif score < 2.0:
                label = "WEAK"
                colour = "#FFD54F"
            else:
                label = "ACTIVE"
                colour = "#A5D6A7"

            dim_col = DIM_COLOURS[dim_i % len(DIM_COLOURS)]

            ax.plot(N_SWEEP, means, color=dim_col, linewidth=1.8, zorder=3)
            ax.fill_between(N_SWEEP, means - 2 * stds, means + 2 * stds,
                            color=dim_col, alpha=0.15, zorder=2)

            # Mark x_star position
            ax.axvline(x_star[dim_i], color="white", linewidth=0.8,
                       linestyle="--", alpha=0.5, zorder=4)

            # Score annotation
            ax.text(0.98, 0.88, f"S={score:.2f} {label}",
                    transform=ax.transAxes, fontsize=6.5,
                    color=colour, ha="right", va="top", fontweight="bold")
            ax.text(0.02, 0.88, f"d{dim_i+1}",
                    transform=ax.transAxes, fontsize=6.5,
                    color=dim_col, ha="left", va="top", fontweight="bold")

            ax.tick_params(colors="#aaaaaa", labelsize=5.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#333333")

            # Only show x-label on bottom subplot
            if dim_i < d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("dim value", fontsize=6.5, color="#888888")

            assessment = label
            print(f"  {FUNC_NAMES[fi]:<22} d{dim_i+1:<4} "
                  f"  score={score:>8.3f}   {assessment}")

        # Function title on top subplot
        first_ax = fig.axes[-(d)]   # approximate
        row_axes = [ax for ax in fig.axes if ax.get_subplotspec().get_gridspec() is inner]
        if row_axes:
            row_axes[0].set_title(FUNC_NAMES[fi], fontsize=8,
                                  color="white", pad=3)

        print()

    print("=" * 72)

    # ── Sensitivity summary ───────────────────────────────────────────────────
    print("\nDimension Sensitivity Summary (S < 0.5 = potentially irrelevant):")
    for fi in range(1, 9):
        d = DIMS[fi]
        scores = all_scores[fi]
        irrelevant = [f"d{i+1}(S={scores[i]:.2f})"
                      for i in range(d) if scores[i] < 0.5]
        weak       = [f"d{i+1}(S={scores[i]:.2f})"
                      for i in range(d) if 0.5 <= scores[i] < 2.0]
        active     = [f"d{i+1}(S={scores[i]:.2f})"
                      for i in range(d) if scores[i] >= 2.0]

        print(f"\n  {FUNC_NAMES[fi]}")
        if active:
            print(f"    ACTIVE     : {', '.join(active)}")
        if weak:
            print(f"    WEAK       : {', '.join(weak)}")
        if irrelevant:
            print(f"    IRRELEVANT?: {', '.join(irrelevant)}")
            print(f"    --> Oracle test: fix active dims at x*, vary "
                  f"{irrelevant[0].split('(')[0]} to [0.1, 0.9] in a future query")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(tracker_dir,
                            f"dim_sensitivity_week{latest_week}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
