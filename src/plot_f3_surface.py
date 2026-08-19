"""
plot_f3_surface.py
==================
Generates a GP surrogate surface plot for F3 (3D Drug Discovery).
Since F3 is 3-dimensional, we produce three 2D cross-section slices:
  - d1 vs d2 (d3 fixed at best known point)
  - d1 vs d3 (d2 fixed at best known point)
  - d2 vs d3 (d1 fixed at best known point)

Also overlays all actual queried data points on each slice.
"""

import os
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(OUTPUT_DIR, "next_week_data", "function_3")
WEEKS_DIRS  = {
    1: os.path.join(OUTPUT_DIR, "results_week1"),
    2: os.path.join(OUTPUT_DIR, "results_week2"),
    3: os.path.join(OUTPUT_DIR, "results_week3"),
}
FUNC_IDX    = 3   # F3 is index 3 (1-based), position 2 (0-based) in txt files
RESOLUTION  = 80  # grid points per axis (higher = sharper but slower)

DIM_LABELS  = ["d1", "d2", "d3"]

# ── Load all F3 data ───────────────────────────────────────────────────────────
def load_all_data():
    """Load directly from the accumulated npy files in next_week_data/.
    bbo_optimize.py appends each week's oracle returns into these files,
    so they always contain the complete dataset up to the latest week run.
    """
    X = np.load(os.path.join(DATA_DIR, "initial_inputs.npy"))
    y = np.load(os.path.join(DATA_DIR, "initial_outputs.npy"))
    return X, y


# ── Build kernel (same as bbo_optimize.py) ─────────────────────────────────────
def build_kernel():
    return (ConstantKernel(1.0) *
            Matern(length_scale=np.ones(3),
                   length_scale_bounds=(1e-2, 1e2), nu=2.5))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading F3 data...")
    X, y = load_all_data()
    print(f"  Total points: {len(y)} | Best y = {np.max(y):.6f}")
    print(f"  Best x: {X[np.argmax(y)]}")

    print("\nFitting GP surrogate...")
    gp = GaussianProcessRegressor(
        kernel=build_kernel(), alpha=1e-5,
        normalize_y=True, n_restarts_optimizer=10, random_state=42)
    gp.fit(X, y)
    print(f"  Fitted kernel: {gp.kernel_}")

    # Best known point — used to fix the "third" dimension in each slice
    x_star = X[np.argmax(y)]
    print(f"\nBest known point (x*): {x_star}")

    # Grid
    g = np.linspace(0, 1, RESOLUTION)

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 7))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        "F3 — 3D Drug Discovery: GP Surrogate Surface (2D Cross-Sections)\n"
        "Third dimension fixed at best known point  |  ✕ = queried points  |  ★ = best known",
        fontsize=13, fontweight="bold", color="white", y=1.02)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Pairs: (x-axis dim, y-axis dim, fixed dim)
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    titles = [
        f"d1 vs d2  (d3 fixed @ {x_star[2]:.3f})",
        f"d1 vs d3  (d2 fixed @ {x_star[1]:.3f})",
        f"d2 vs d3  (d1 fixed @ {x_star[0]:.3f})",
    ]

    global_min, global_max = None, None

    # Pre-compute all grids to get consistent colour scale
    grids = []
    for xi, yi, zi in pairs:
        G1, G2 = np.meshgrid(g, g)
        test_pts = np.tile(x_star, (RESOLUTION * RESOLUTION, 1))
        test_pts[:, xi] = G1.ravel()
        test_pts[:, yi] = G2.ravel()
        mu, _ = gp.predict(test_pts, return_std=True)
        Z = mu.reshape(RESOLUTION, RESOLUTION)
        grids.append((G1, G2, Z))
        if global_min is None:
            global_min, global_max = Z.min(), Z.max()
        else:
            global_min = min(global_min, Z.min())
            global_max = max(global_max, Z.max())

    norm = Normalize(vmin=global_min, vmax=global_max)
    cmap = plt.get_cmap("plasma")

    for idx, ((xi, yi, zi), (G1, G2, Z), title) in enumerate(
            zip(pairs, grids, titles)):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor("#111111")

        # Surface
        cf = ax.contourf(G1, G2, Z, levels=40, cmap=cmap, norm=norm)
        ax.contour(G1, G2, Z, levels=10, colors="white", alpha=0.15,
                   linewidths=0.5)

        # Scatter all data points (projected onto this slice)
        ax.scatter(X[:, xi], X[:, yi], c=y, cmap=cmap, norm=norm,
                   edgecolors="white", linewidths=0.6, s=60,
                   zorder=5, label="Queried")

        # Mark best point
        ax.scatter(x_star[xi], x_star[yi], marker="*", s=280,
                   color="#FFD700", edgecolors="white", linewidths=0.8,
                   zorder=6, label="Best")

        # Mark Week 4 GP suggestion (boundary-pinned point)
        w4_gp = np.array([0.187086, 0.000001, 0.000001])
        ax.scatter(w4_gp[xi], w4_gp[yi], marker="D", s=100,
                   color="#00E5FF", edgecolors="white", linewidths=0.6,
                   zorder=6, label="W4 GP suggestion")

        ax.set_xlabel(DIM_LABELS[xi], fontsize=10, color="#aaaaaa")
        ax.set_ylabel(DIM_LABELS[yi], fontsize=10, color="#aaaaaa")
        ax.set_title(title, fontsize=9, color="white", pad=6)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines[["top", "right", "left", "bottom"]].set_color("#444444")
        ax.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#444444",
                  loc="upper right")
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            colors="#aaaaaa", labelsize=7)

    out_path = os.path.join(OUTPUT_DIR, "progress_tracker_week_3",
                            "f3_surface_week3.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
