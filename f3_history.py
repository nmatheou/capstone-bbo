"""
f3_history.py
=============
Prints a history table for F3 showing:
  - All initial baseline points (LHS, no prior prediction)
  - Each weekly query: what the GP predicted vs what the oracle returned
"""
import os, ast, re
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

OUTPUT_DIR = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/antigravity"
DATA_DIR   = os.path.join(OUTPUT_DIR, "next_week_data", "function_3")
WEEKS_DIRS = {
    1: os.path.join(OUTPUT_DIR, "results_week1"),
    2: os.path.join(OUTPUT_DIR, "results_week2"),
    3: os.path.join(OUTPUT_DIR, "results_week3"),
}
FUNC_IDX = 3  # 1-based

def build_kernel():
    return (ConstantKernel(1.0) *
            Matern(length_scale=np.ones(3),
                   length_scale_bounds=(1e-2, 1e2), nu=2.5))

def fit_gp(X, y):
    gp = GaussianProcessRegressor(
        kernel=build_kernel(), alpha=1e-5,
        normalize_y=True, n_restarts_optimizer=10, random_state=42)
    gp.fit(X, y)
    return gp

def parse_week(wdir):
    inp_path = os.path.join(wdir, "inputs.txt")
    out_path = os.path.join(wdir, "outputs.txt")
    with open(inp_path) as f:
        s = f.read().replace("array(", "").replace(")", "")
    s = re.sub(r'\]\s*\n\s*\[', '],\n[', s)
    parsed = ast.literal_eval(f"[{s}]")
    x = np.array(parsed[-1][FUNC_IDX - 1])

    with open(out_path) as f:
        s2 = f.read()
    s2 = re.sub(r'\]\s*\n\s*\[', '],\n[', s2)
    out_parsed = ast.literal_eval(f"[{s2}]")
    y = float(out_parsed[-1][FUNC_IDX - 1])
    return x, y

# ── Load baseline data ─────────────────────────────────────────────────────────
X_base = np.load(os.path.join(DATA_DIR, "initial_inputs.npy"))
y_base = np.load(os.path.join(DATA_DIR, "initial_outputs.npy"))

# The baseline npy now includes weekly points appended by bbo_optimize
# We need to separate: first N_BASE rows = LHS baseline, rest = weekly queries
# Count by checking which rows match weekly query inputs
weekly_queries = {}
for week, wdir in sorted(WEEKS_DIRS.items()):
    if os.path.exists(os.path.join(wdir, "inputs.txt")):
        x_q, y_q = parse_week(wdir)
        weekly_queries[week] = (x_q, y_q)

# Find which rows in X_base are weekly queries
weekly_row_indices = {}
for week, (x_q, _) in weekly_queries.items():
    for i, row in enumerate(X_base):
        if np.allclose(row, x_q, atol=1e-5):
            weekly_row_indices[week] = i
            break

# Baseline rows = all rows NOT in weekly queries
weekly_rows = set(weekly_row_indices.values())
baseline_mask = [i for i in range(len(X_base)) if i not in weekly_rows]
X_lhs = X_base[baseline_mask]
y_lhs = y_base[baseline_mask]

# ── Print table ────────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'#':<4} {'Source':<10} {'d1':>8} {'d2':>8} {'d3':>8}  {'GP Pred':>14} {'GP Std':>12} {'Oracle y':>12}  {'Delta (pred-act)':>16}")
print("=" * 100)

# Print LHS baseline (no GP prediction available — these are the training seed)
for i, (x, y) in enumerate(zip(X_lhs, y_lhs)):
    print(f"{i+1:<4} {'LHS'::<10} {x[0]:>8.4f} {x[1]:>8.4f} {x[2]:>8.4f}  {'—':>14} {'—':>12} {y:>12.6f}  {'—':>16}")

print("-" * 100)

# For each weekly query, fit GP on all data BEFORE that query and predict
X_known = X_lhs.copy()
y_known = y_lhs.copy()

for week in sorted(weekly_queries.keys()):
    x_q, y_oracle = weekly_queries[week]

    # Fit GP on all data available BEFORE this week's query
    gp = fit_gp(X_known, y_known)
    mu, sigma = gp.predict(x_q.reshape(1, -1), return_std=True)
    mu    = mu[0]
    sigma = sigma[0]
    delta = mu - y_oracle

    row_num = len(baseline_mask) + week
    print(f"{row_num:<4} {'W' + str(week) + ' Query':<10} {x_q[0]:>8.4f} {x_q[1]:>8.4f} {x_q[2]:>8.4f}  {mu:>14.6f} {sigma:>12.6f} {y_oracle:>12.6f}  {delta:>+16.6f}")

    # Add this point to known data for next week's GP
    X_known = np.vstack([X_known, x_q.reshape(1, -1)])
    y_known = np.append(y_known, y_oracle)

print("=" * 100)
print(f"\nBaseline best y = {np.max(y_lhs):.6f} at x = {X_lhs[np.argmax(y_lhs)]}")
print(f"Overall best y  = {np.max(y_base):.6f} at x = {X_base[np.argmax(y_base)]}")
