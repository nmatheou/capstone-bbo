"""
length_scale_tracker.py
========================
Re-fits the GP surrogate on each week's accumulated data and reports how the
per-dimension Matern ARD length scales evolve over time.

Stable length scales => GP has learnt the landscape => strategy is converging
Volatile length scales => GP still uncertain => exploration remains justified

Outputs
-------
1. Console table: per-function, per-week, per-dimension length scales
2. length_scale_dashboard.png  -- one subplot per function, x=week, y=length scale per dim
"""

import sys
import re
import ast
import os
import numpy as np
import matplotlib.pyplot as plt
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
    1: "F1 2D Sparse Needle",
    2: "F2 2D Noisy Sim",
    3: "F3 3D Drug Discovery",
    4: "F4 4D Warehouse",
    5: "F5 4D Chemical Yield",
    6: "F6 5D Cake Composite",
    7: "F7 6D ML Hyperparams",
    8: "F8 8D ML Hyperparams",
}

DIM_COLOURS = [
    "#00E5FF", "#FFD54F", "#A5D6A7", "#FF7043",
    "#CE93D8", "#80DEEA", "#FFCC80", "#F48FB1"
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_week_query(week_dir):
    """Return (inputs_list, outputs_list) from a week results directory."""
    try:
        with open(os.path.join(week_dir, "inputs.txt"), "r") as f:
            raw = f.read().replace("array(", "").replace(")", "")
        raw = re.sub(r'\]\s*\n\s*\[', '],\n[', raw)
        inputs_list = ast.literal_eval(f"[{raw}]")[-1]

        with open(os.path.join(week_dir, "outputs.txt"), "r") as f:
            out_raw = f.read()
        out_raw = re.sub(r'\]\s*\n\s*\[', '],\n[', out_raw)
        outputs_list = ast.literal_eval(f"[{out_raw}]")[-1]

        return inputs_list, outputs_list
    except FileNotFoundError:
        return None, None


def build_kernel(func_idx, d):
    """Return the same kernel used in the optimization scripts."""
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


def extract_length_scales(gp, d):
    """Pull the fitted ARD length scales out of a fitted GP kernel."""
    # Walk the kernel tree to find the Matern component
    kernel = gp.kernel_
    for part in [kernel] + list(getattr(kernel, 'get_params', lambda: {})().values()):
        if isinstance(part, Matern):
            ls = np.atleast_1d(part.length_scale)
            if len(ls) == d:
                return ls
    # Fallback: traverse k1/k2
    for attr in ['k1', 'k2']:
        sub = getattr(kernel, attr, None)
        if sub is None:
            continue
        for attr2 in ['k1', 'k2']:
            sub2 = getattr(sub, attr2, None)
            if isinstance(sub2, Matern):
                ls = np.atleast_1d(sub2.length_scale)
                if len(ls) == d:
                    return ls
            if isinstance(sub, Matern):
                ls = np.atleast_1d(sub.length_scale)
                if len(ls) == d:
                    return ls
    return np.full(d, np.nan)


def coefficient_of_variation(arr):
    """CV = std/mean — a scale-invariant volatility measure."""
    arr = np.array(arr)
    if arr.mean() == 0 or np.isnan(arr).all():
        return np.nan
    return np.std(arr) / np.abs(np.mean(arr))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Build week-by-week accumulated datasets: {func_idx: {week: (X, y)}}
    snapshots = {}

    for fi in range(1, 9):
        d = DIMS[fi]
        f_dir = os.path.join(BASE_DIR, f"function_{fi}")
        X0 = np.load(os.path.join(f_dir, "initial_inputs.npy"))
        y0 = np.load(os.path.join(f_dir, "initial_outputs.npy"))

        snapshots[fi] = {0: (X0.copy(), y0.copy())}
        X_acc, y_acc = X0.copy(), y0.copy()

        for week, week_dir in WEEKS_DIRS.items():
            inputs_list, outputs_list = load_week_query(week_dir)
            if inputs_list is None:
                break
            new_x = np.atleast_2d(np.array(inputs_list[fi - 1]))
            new_y = outputs_list[fi - 1]
            X_acc = np.vstack([X_acc, new_x])
            y_acc = np.append(y_acc, new_y)
            snapshots[fi][week] = (X_acc.copy(), y_acc.copy())

    # Fit GPs and extract length scales
    # length_scales[fi][week] = np.array of shape (d,)
    length_scales = {}
    available_weeks = []

    print("\nFitting GPs on each snapshot (this takes a moment)...")
    for fi in range(1, 9):
        d = DIMS[fi]
        length_scales[fi] = {}
        for week, (X, y) in sorted(snapshots[fi].items()):
            kernel = build_kernel(fi, d)
            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-5,
                normalize_y=True, n_restarts_optimizer=10, random_state=42)
            gp.fit(X, y)
            ls = extract_length_scales(gp, d)
            length_scales[fi][week] = ls
            if week not in available_weeks:
                available_weeks.append(week)
        print(f"  F{fi} done ({len(snapshots[fi])} snapshots)")

    available_weeks = sorted(available_weeks)
    fitted_weeks = sorted(length_scales[1].keys())

    # ── Console Table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Function':<22} {'Week':<6} {'Length Scales (per dim)':>40}  {'CV (volatility)':>16}")
    print("=" * 90)

    for fi in range(1, 9):
        d = DIMS[fi]
        all_ls = []
        for week in fitted_weeks:
            ls = length_scales[fi].get(week)
            if ls is not None:
                ls_str = "  ".join(f"d{i+1}={v:.4f}" for i, v in enumerate(ls))
                all_ls.append(ls)
                print(f"{FUNC_NAMES[fi]:<22} W{week:<5} {ls_str}")
        # Print per-dim CV across weeks
        all_ls = np.array(all_ls)
        cv_str = "  ".join(
            f"d{i+1}={coefficient_of_variation(all_ls[:, i]):.3f}"
            for i in range(d))
        stability = "STABLE" if all(
            coefficient_of_variation(all_ls[:, i]) < 0.3 for i in range(d)
        ) else "VOLATILE"
        print(f"  --> CV across weeks: {cv_str}  [{stability}]")
        print()

    print("=" * 90)

    # ── Dashboard Plot ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("GP Length Scale Evolution per Function & Dimension",
                 fontsize=14, fontweight="bold", color="white", y=1.01)
    fig.patch.set_facecolor("#0d0d0d")

    for idx, fi in enumerate(range(1, 9)):
        ax = axes[idx // 4][idx % 4]
        ax.set_facecolor("#111111")
        d = DIMS[fi]

        weeks_with_data = sorted(length_scales[fi].keys())
        x_ticks = weeks_with_data

        for dim_i in range(d):
            ys = [length_scales[fi][w][dim_i] for w in weeks_with_data]
            colour = DIM_COLOURS[dim_i % len(DIM_COLOURS)]
            ax.plot(weeks_with_data, ys, marker="o", markersize=6,
                    linewidth=2, color=colour, label=f"d{dim_i+1}", zorder=3)
            # Shade between min and max to show spread
            if len(ys) > 1:
                ax.fill_between(weeks_with_data,
                                [min(ys)] * len(ys), [max(ys)] * len(ys),
                                color=colour, alpha=0.06)

        # Stability annotation
        all_ls = np.array([length_scales[fi][w] for w in weeks_with_data])
        max_cv = max(coefficient_of_variation(all_ls[:, i]) for i in range(d)
                     if not np.isnan(coefficient_of_variation(all_ls[:, i])))
        stability_txt = "STABLE" if max_cv < 0.3 else f"VOLATILE (CV={max_cv:.2f})"
        colour_txt = "#A5D6A7" if max_cv < 0.3 else "#FF7043"
        ax.text(0.97, 0.97, stability_txt,
                transform=ax.transAxes, fontsize=8,
                color=colour_txt, ha="right", va="top", fontweight="bold")

        ax.set_title(FUNC_NAMES[fi], fontsize=9, color="white", pad=4)
        ax.set_xlabel("Week", fontsize=8, color="#aaaaaa")
        ax.set_ylabel("Length Scale", fontsize=8, color="#aaaaaa")
        ax.set_xticks(x_ticks)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#444444")
        ax.grid(axis="y", color="#333333", linewidth=0.5, linestyle="--")

        ax.legend(fontsize=7, facecolor="#1a1a1a",
                  edgecolor="#444444", loc="lower right")

    # Save
    completed_weeks = [w for w in fitted_weeks if w > 0]
    latest_week = completed_weeks[-1] if completed_weeks else 0
    tracker_dir = os.path.join(OUTPUT_DIR,
                               f"results/week_{latest_week}/progress")
    os.makedirs(tracker_dir, exist_ok=True)
    out_path = os.path.join(tracker_dir,
                            f"length_scale_dashboard_week{latest_week}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"\nSaved: {out_path}")

    # ── Stability Summary ─────────────────────────────────────────────────────
    print("\nStability Summary (CV < 0.30 = STABLE, >= 0.30 = VOLATILE):")
    for fi in range(1, 9):
        d = DIMS[fi]
        all_ls = np.array([length_scales[fi][w] for w in fitted_weeks])
        cvs = [coefficient_of_variation(all_ls[:, i]) for i in range(d)]
        max_cv = max(v for v in cvs if not np.isnan(v))
        stability = "STABLE  " if max_cv < 0.3 else "VOLATILE"
        dim_cvs = "  ".join(f"d{i+1}:{v:.3f}" for i, v in enumerate(cvs))
        print(f"  F{fi} [{stability}]  max CV={max_cv:.3f}  |  {dim_cvs}")


if __name__ == "__main__":
    main()
