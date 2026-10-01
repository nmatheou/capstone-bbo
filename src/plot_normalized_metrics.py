"""
plot_normalized_metrics.py
==========================
Generates a normalized metrics dashboard across weeks 1 to N,
scaling each function relative to its max absolute value to avoid
visual domination by high-magnitude functions like F5.
"""

import sys
import numpy as np
import os
import ast
import re
import matplotlib.pyplot as plt
import warnings
import argparse

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
plt.style.use("dark_background")

parser = argparse.ArgumentParser()
parser.add_argument('--week', type=int, required=True, help="Target week (e.g. 6)")
args, unknown = parser.parse_known_args()
TARGET_WEEK = args.week

BASE_DIR   = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/M12/Data/M12"
OUTPUT_DIR = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/antigravity"
WEEKS_DIRS = {w: os.path.join(OUTPUT_DIR, "data", f"week_{w}_returns") for w in range(1, TARGET_WEEK + 1)}
RESULTS_FILES = {w: os.path.join(OUTPUT_DIR, "submissions", f"week_{w}", f"results_week{w}.txt" if w < 5 else "suggestions.txt") for w in range(1, TARGET_WEEK + 1)}

FUNC_NAMES = {1: "F1", 2: "F2", 3: "F3", 4: "F4", 5: "F5", 6: "F6", 7: "F7", 8: "F8"}

def load_week_query(week_dir):
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
    results = {}
    try:
        with open(filepath, "r") as f:
            content = f.read()
        blocks = re.split(r"--- Function (\d+) ---", content)
        for i in range(1, len(blocks), 2):
            func_idx = int(blocks[i])
            block    = blocks[i + 1]
            m  = re.search(r"GP Std:\s*([\d.eE+]+)", block)
            if m:
                results[func_idx] = {"gp_std": float(m.group(1))}
    except FileNotFoundError:
        pass
    return results

def build_history(func_idx):
    f_dir = os.path.join(BASE_DIR, f"function_{func_idx}")
    initial_y = np.load(os.path.join(f_dir, "initial_outputs.npy"))
    weekly_y = {}
    for week, week_dir in WEEKS_DIRS.items():
        _, outputs = load_week_query(week_dir)
        if outputs is not None:
            weekly_y[week] = outputs[func_idx - 1]
    return initial_y, weekly_y

all_results = {w: parse_results_txt(f) for w, f in RESULTS_FILES.items()}
histories = {}
best_by_week = {}
delta = {}
gp_std = {}

for fi in range(1, 9):
    init_y, wk_y = build_history(fi)
    histories[fi] = (init_y, wk_y)
    running_best = np.max(init_y)
    best_by_week[fi] = {0: running_best}
    delta[fi] = {}
    gp_std[fi] = {}
    for w in sorted(wk_y.keys()):
        prev_best = running_best
        running_best = max(running_best, wk_y[w])
        best_by_week[fi][w] = running_best
        delta[fi][w] = running_best - prev_best
    for w, res in all_results.items():
        if fi in res:
            gp_std[fi][w] = res[fi]["gp_std"]

completed_weeks = sorted({w for fi in range(1, 9) for w in best_by_week[fi] if w > 0})
latest_week = completed_weeks[-1] if completed_weeks else 0
available_weeks = completed_weeks
tracker_dir = os.path.join(OUTPUT_DIR, f"results/week_{TARGET_WEEK}/progress")
os.makedirs(tracker_dir, exist_ok=True)

norm_best = {}
norm_delta = {}
for fi in range(1, 9):
    max_val = max(abs(best_by_week[fi][w]) for w in best_by_week[fi])
    if max_val == 0: max_val = 1
    norm_best[fi] = {w: v / max_val for w, v in best_by_week[fi].items()}
    
    max_d = max(abs(delta[fi][w]) for w in delta[fi])
    if max_d == 0: max_d = 1
    norm_delta[fi] = {w: v / max_d for w, v in delta[fi].items()}

fig2, axes2 = plt.subplots(1, 3, figsize=(22, 6))
fig2.suptitle(f"BBO Metrics Overview (NORMALIZED) — Weeks 0 to {latest_week}", fontsize=14, fontweight="bold", color="white")
fig2.patch.set_facecolor("#0d0d0d")

func_labels  = [f"F{i}" for i in range(1, 9)]
bar_width    = 0.8 / max(len(available_weeks) + 1, 1)
week_palette = ["#4FC3F7", "#FFD54F", "#A5D6A7", "#FF7043", "#D1C4E9", "#FF80AB", "#80D8FF"]
all_weeks = [0] + available_weeks

# Panel 1
ax = axes2[0]
ax.set_facecolor("#111111")
for wi, w in enumerate(all_weeks):
    offsets = np.arange(8) + (wi - len(all_weeks) / 2) * bar_width
    vals    = [norm_best[fi].get(w, np.nan) for fi in range(1, 9)]
    ax.bar(offsets, vals, width=bar_width * 0.9, color=week_palette[wi % len(week_palette)], label=f"W{w}" if w>0 else "Init", alpha=0.85)
ax.set_title("Best Observed Y (Relative to F-Max)", color="white", fontsize=11)
ax.set_xticks(np.arange(8))
ax.set_xticklabels(func_labels, color="#aaaaaa", fontsize=9)
ax.tick_params(colors="#aaaaaa")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#444444")
ax.grid(axis="y", color="#333333", linewidth=0.5, linestyle="--")
ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#444444")

# Panel 2
ax2 = axes2[1]
ax2.set_facecolor("#111111")
for wi, w in enumerate(available_weeks):
    offsets = np.arange(8) + (wi - len(available_weeks) / 2) * bar_width
    vals    = [norm_delta[fi].get(w, 0.0) for fi in range(1, 9)]
    colours = [week_palette[(wi + 1) % len(week_palette)] if v >= 0 else "#EF9A9A" for v in vals]
    ax2.bar(offsets, vals, width=bar_width * 0.9, color=colours, alpha=0.85, label=f"W{w}")
ax2.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
ax2.set_title("Improvement Δy per Week (Relative to F-Max Δ)", color="white", fontsize=11)
ax2.set_xticks(np.arange(8))
ax2.set_xticklabels(func_labels, color="#aaaaaa", fontsize=9)
ax2.tick_params(colors="#aaaaaa")
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#444444")
ax2.grid(axis="y", color="#333333", linewidth=0.5, linestyle="--")
ax2.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#444444")

# Panel 3
ax3 = axes2[2]
ax3.set_facecolor("#111111")
norm_std = {}
for fi in range(1, 9):
    max_s = max([abs(gp_std[fi][w]) for w in gp_std[fi]] + [0.001])
    norm_std[fi] = {w: v / max_s for w, v in gp_std[fi].items()}

for wi, w in enumerate(available_weeks):
    offsets = np.arange(8) + (wi - len(available_weeks) / 2) * bar_width
    vals    = [norm_std[fi].get(w, 0.0) for fi in range(1, 9)]
    ax3.bar(offsets, vals, width=bar_width * 0.9, color=week_palette[(wi + 1) % len(week_palette)], alpha=0.85, label=f"W{w}")
ax3.set_title("GP Std at Suggested Point (Relative to Max Std)", color="white", fontsize=11)
ax3.set_xticks(np.arange(8))
ax3.set_xticklabels(func_labels, color="#aaaaaa", fontsize=9)
ax3.tick_params(colors="#aaaaaa")
ax3.spines[["top", "right"]].set_visible(False)
ax3.spines[["left", "bottom"]].set_color("#444444")
ax3.grid(axis="y", color="#333333", linewidth=0.5, linestyle="--")
ax3.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#444444")

plt.tight_layout()
out_path = os.path.join(tracker_dir, f"metrics_dashboard_normalized_week{latest_week}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
plt.close()
print(f"Saved: {out_path}")
