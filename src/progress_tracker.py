"""
progress_tracker.py
====================
Tracks and visualises Bayesian Optimisation progress across weeks for all 8
black-box functions.

Outputs
-------
1. Console summary table  (best y, Δ improvement, GP Std, Acq Score per week)
2. trajectory_dashboard.png  — per-function best-y step plots + query markers
3. metrics_dashboard.png     — grouped bar charts: best y, Δ improvement, GP Std
"""

import sys
import numpy as np
import os
import ast
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

# Force UTF-8 output on Windows terminals
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
RESULTS_FILES = {w: os.path.join(OUTPUT_DIR, "submissions", f"week_{w}", f"results_week{w}.txt" if w < 5 else "suggestions.txt") for w in range(1, TARGET_WEEK + 1)}

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
WEEK_COLOURS = {0: "#888888", 1: "#4FC3F7", 2: "#FFD54F", 3: "#A5D6A7"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_week_query(week_dir):
    """Return (inputs_list, outputs_list) from a week results directory, or
    (None, None) if the files don't exist yet."""
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


def parse_results_txt(filepath):
    """Parse results_weekN.txt → dict {func_idx: {gp_mean, gp_std, acq_score, coord}}"""
    results = {}
    try:
        with open(filepath, "r") as f:
            content = f.read()
        blocks = re.split(r"--- Function (\d+) ---", content)
        for i in range(1, len(blocks), 2):
            func_idx = int(blocks[i])
            block    = blocks[i + 1]
            m  = re.search(
                r"GP Mean:\s*([-\d.eE+]+),\s*GP Std:\s*([\d.eE+]+),\s*Acq Score:\s*([\d.eE+]+)",
                block)
            cm = re.search(r"Coordinate String:\s*([\d.\-]+)", block)
            if m and cm:
                results[func_idx] = {
                    "gp_mean":   float(m.group(1)),
                    "gp_std":    float(m.group(2)),
                    "acq_score": float(m.group(3)),
                    "coord":     cm.group(1),
                }
    except FileNotFoundError:
        pass
    return results


def build_history(func_idx):
    """Build chronological y history for a function.

    Returns
    -------
    initial_y  : np.ndarray   — initial data outputs
    weekly_y   : dict         — {week: actual returned y value} for each completed week
    """
    # Initial data
    f_dir = os.path.join(BASE_DIR, f"function_{func_idx}")
    initial_y = np.load(os.path.join(f_dir, "initial_outputs.npy"))

    weekly_y = {}
    for week, week_dir in WEEKS_DIRS.items():
        _, outputs = load_week_query(week_dir)
        if outputs is not None:
            weekly_y[week] = outputs[func_idx - 1]

    return initial_y, weekly_y


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Collect all data ──────────────────────────────────────────────────────
    all_results = {w: parse_results_txt(f) for w, f in RESULTS_FILES.items()}

    histories    = {}   # func_idx → (initial_y, weekly_y)
    best_by_week = {}   # func_idx → {0: initial_best, 1: best_after_w1, ...}
    delta        = {}   # func_idx → {1: Δ after w1, 2: Δ after w2, ...}
    gp_std       = {}   # func_idx → {week: gp_std}
    acq_score    = {}   # func_idx → {week: acq_score}

    for fi in range(1, 9):
        init_y, wk_y = build_history(fi)
        histories[fi] = (init_y, wk_y)

        running_best = np.max(init_y)
        best_by_week[fi] = {0: running_best}
        delta[fi]        = {}
        gp_std[fi]       = {}
        acq_score[fi]    = {}

        for w in sorted(wk_y.keys()):
            prev_best    = running_best
            running_best = max(running_best, wk_y[w])
            best_by_week[fi][w] = running_best
            delta[fi][w]        = running_best - prev_best

        for w, res in all_results.items():
            if fi in res:
                gp_std[fi][w]    = res[fi]["gp_std"]
                acq_score[fi][w] = res[fi]["acq_score"]

    # ── Console Table ─────────────────────────────────────────────────────────
    # Determine latest completed week for subfolder naming
    completed_weeks = sorted(
        {w for fi in range(1, 9) for w in best_by_week[fi] if w > 0})
    latest_week = completed_weeks[-1] if completed_weeks else 0
    tracker_dir = os.path.join(OUTPUT_DIR, f"results/week_{latest_week}/progress")
    os.makedirs(tracker_dir, exist_ok=True)
    print(f"Saving outputs to: {tracker_dir}\n")
    available_weeks = completed_weeks

    header = f"{'Function':<24} {'Init Best':>12}"
    for w in available_weeks:
        header += f"  {'W'+str(w)+' Best':>12}  {'D W'+str(w):>10}  {'GP Std W'+str(w):>11}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for fi in range(1, 9):
        row = f"{FUNC_NAMES[fi]:<24} {best_by_week[fi][0]:>12.6f}"
        for w in available_weeks:
            b = best_by_week[fi].get(w, float("nan"))
            d = delta[fi].get(w, float("nan"))
            s = gp_std[fi].get(w, float("nan"))
            row += f"  {b:>12.6f}  {d:>+10.6f}  {s:>11.6f}"
        print(row)

    print("=" * len(header) + "\n")

    # ── Plot 1: Trajectory Dashboard ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("BBO Optimization Progress — Best Observed Y per Function",
                 fontsize=15, fontweight="bold", color="white", y=1.01)

    all_weeks = [0] + available_weeks

    for idx, fi in enumerate(range(1, 9)):
        ax = axes[idx // 4][idx % 4]
        init_y, wk_y = histories[fi]

        # Build cumulative best step data
        x_pts = [0]
        y_pts = [np.max(init_y)]
        for w in sorted(wk_y.keys()):
            x_pts.append(w)
            y_pts.append(max(y_pts[-1], wk_y[w]))

        # Shade exploration phase (weeks 1-3)
        ax.axvspan(-0.5, 3.5, alpha=0.07, color="#4FC3F7")

        # Step line — cumulative best
        ax.step(x_pts, y_pts, where="post", color="#00E5FF",
                linewidth=2.5, zorder=3, label="Cumulative Best")

        # Initial data distribution (violin-style scatter)
        jitter = np.random.uniform(-0.12, 0.12, len(init_y))
        ax.scatter(np.zeros(len(init_y)) + jitter, init_y,
                   color=WEEK_COLOURS[0], alpha=0.5, s=18, zorder=2,
                   label="Initial Data")

        # Weekly query markers
        for w, y_val in wk_y.items():
            colour = WEEK_COLOURS.get(w, "#FF7043")
            ax.scatter(w, y_val, color=colour, s=120, zorder=5,
                       edgecolors="white", linewidths=0.8,
                       marker="*", label=f"W{w} Query")
            ax.annotate(f"{y_val:.4f}", (w, y_val),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color=colour)

        ax.set_title(FUNC_NAMES[fi], fontsize=9, color="white", pad=4)
        ax.set_xlabel("Week", fontsize=8, color="#aaaaaa")
        ax.set_ylabel("Best y", fontsize=8, color="#aaaaaa")
        ax.set_xticks(all_weeks)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#444444")
        ax.grid(axis="y", color="#333333", linewidth=0.5, linestyle="--")

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color="#00E5FF", linewidth=2, label="Cumulative Best"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=WEEK_COLOURS[0],
               markersize=6, label="Initial Data", linestyle="None"),
    ]
    for w in available_weeks:
        legend_elements.append(
            Line2D([0], [0], marker="*", color="w",
                   markerfacecolor=WEEK_COLOURS.get(w, "#FF7043"),
                   markersize=10, label=f"W{w} Query Return", linestyle="None"))

    fig.legend(handles=legend_elements, loc="lower center", ncol=len(legend_elements),
               fontsize=8, facecolor="#1a1a1a", edgecolor="#444444",
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    traj_path = os.path.join(tracker_dir, f"trajectory_dashboard_week{latest_week}.png")
    plt.savefig(traj_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"Saved: {traj_path}")

    # ── Plot 2: Metrics Dashboard ─────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle("BBO Metrics Overview — All Functions & Weeks",
                  fontsize=14, fontweight="bold", color="white")
    fig2.patch.set_facecolor("#0d0d0d")

    func_labels  = [f"F{i}" for i in range(1, 9)]
    bar_width    = 0.8 / max(len(available_weeks), 1)
    week_palette = ["#4FC3F7", "#FFD54F", "#A5D6A7", "#FF7043"]

    # — Panel 1: Best Y per function per week ——————————————————————————————————
    ax = axes2[0]
    ax.set_facecolor("#111111")
    for wi, w in enumerate(all_weeks):
        offsets = np.arange(8) + (wi - len(all_weeks) / 2) * bar_width
        vals    = [best_by_week[fi].get(w, np.nan) for fi in range(1, 9)]
        bars    = ax.bar(offsets, vals, width=bar_width * 0.9,
                         color=week_palette[wi % len(week_palette)],
                         label=f"Week {w} ('Initial' if w=0 else '')",
                         alpha=0.85)
    ax.set_title("Best Observed Y", color="white", fontsize=11)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(func_labels, color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444444")
    ax.grid(axis="y", color="#333333", linewidth=0.5, linestyle="--")
    ax.legend([f"Week {w}" if w > 0 else "Initial" for w in all_weeks],
              fontsize=8, facecolor="#1a1a1a", edgecolor="#444444")

    # — Panel 2: Δ Improvement per week ———————————————————————————————————————
    ax2 = axes2[1]
    ax2.set_facecolor("#111111")
    for wi, w in enumerate(available_weeks):
        offsets = np.arange(8) + (wi - len(available_weeks) / 2) * bar_width
        vals    = [delta[fi].get(w, 0.0) for fi in range(1, 9)]
        colours = [week_palette[(wi + 1) % len(week_palette)]
                   if v >= 0 else "#EF9A9A" for v in vals]
        ax2.bar(offsets, vals, width=bar_width * 0.9,
                color=colours, alpha=0.85, label=f"Δ Week {w}")
    ax2.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax2.set_title("Improvement Δy per Week", color="white", fontsize=11)
    ax2.set_xticks(np.arange(8))
    ax2.set_xticklabels(func_labels, color="#aaaaaa", fontsize=9)
    ax2.tick_params(colors="#aaaaaa")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color("#444444")
    ax2.grid(axis="y", color="#333333", linewidth=0.5, linestyle="--")
    ax2.legend([f"Δ W{w}" for w in available_weeks],
               fontsize=8, facecolor="#1a1a1a", edgecolor="#444444")

    # — Panel 3: GP Std trend (proxy for surrogate confidence) ————————————————
    ax3 = axes2[2]
    ax3.set_facecolor("#111111")
    for wi, w in enumerate(available_weeks):
        offsets = np.arange(8) + (wi - len(available_weeks) / 2) * bar_width
        vals    = [gp_std[fi].get(w, np.nan) for fi in range(1, 9)]
        ax3.bar(offsets, vals, width=bar_width * 0.9,
                color=week_palette[(wi + 1) % len(week_palette)],
                alpha=0.85, label=f"GP Std W{w}")
    ax3.set_title("GP Std at Suggested Point\n(lower = more confident)",
                  color="white", fontsize=11)
    ax3.set_xticks(np.arange(8))
    ax3.set_xticklabels(func_labels, color="#aaaaaa", fontsize=9)
    ax3.tick_params(colors="#aaaaaa")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.spines[["left", "bottom"]].set_color("#444444")
    ax3.grid(axis="y", color="#333333", linewidth=0.5, linestyle="--")
    ax3.legend([f"GP Std W{w}" for w in available_weeks],
               fontsize=8, facecolor="#1a1a1a", edgecolor="#444444")

    for ax_ in axes2:
        ax_.set_xlabel("Function", fontsize=9, color="#aaaaaa")

    plt.tight_layout()
    metrics_path = os.path.join(tracker_dir, f"metrics_dashboard_week{latest_week}.png")
    plt.savefig(metrics_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"Saved: {metrics_path}")

    # ── Red Flag Summary ──────────────────────────────────────────────────────
    print("\n[!] Red Flag Check:")
    red_flags = False
    for fi in range(1, 9):
        _, wk_y = histories[fi]
        init_best = best_by_week[fi][0]
        for w in sorted(wk_y.keys()):
            returned_y = wk_y[w]
            if returned_y < init_best:
                print(f"  [WARN] F{fi} Week {w}: query returned {returned_y:.6f} "
                      f"(below initial best {init_best:.6f})")
                red_flags = True
        for w in available_weeks:
            std = gp_std[fi].get(w)
            if std is not None and std < 1e-4:
                print(f"  [WARN] F{fi} Week {w}: GP Std = {std:.6f} - "
                      f"surrogate over-confident, consider widening TR or adding noise")
                red_flags = True
            acq = acq_score[fi].get(w)
            if acq is not None and acq <= 0.0:
                print(f"  [WARN] F{fi} Week {w}: Acq Score = {acq:.6f} - "
                      f"acquisition found no improvement region")
                red_flags = True

    if not red_flags:
        print("  [OK] No red flags detected.\n")


if __name__ == "__main__":
    main()
