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

def run_bbo():
    base_dir = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/M12/Data/M12"
    output_dir = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/antigravity"
    results_file = os.path.join(output_dir, "results_week2.txt")
    dims = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}

    # 1. Parse New Data (Before the loop)
    results_w1_dir = os.path.join(output_dir, "results_week1")
    inputs_file_path = os.path.join(results_w1_dir, "inputs.txt")
    outputs_file_path = os.path.join(results_w1_dir, "outputs.txt")
    
    try:
        with open(inputs_file_path, "r") as f:
            inputs_str = f.read()
        # Clean the string representation of "array(...)"
        inputs_str_clean = inputs_str.replace("array(", "").replace(")", "")
        new_inputs_list = ast.literal_eval(inputs_str_clean)
        
        with open(outputs_file_path, "r") as f:
            outputs_str = f.read()
        new_outputs_list = ast.literal_eval(outputs_str)
    except FileNotFoundError:
        print("Error: inputs.txt or outputs.txt not found.")
        print("Please place these files in:", output_dir)
        return
    except Exception as e:
        print(f"Error parsing inputs/outputs: {e}")
        return

    # Create next_week_data directory
    next_week_dir = os.path.join(output_dir, "next_week_data")
    os.makedirs(next_week_dir, exist_ok=True)

    with open(results_file, 'w') as f_out:
        for func_idx in range(1, 9):
            print(f"--- Function {func_idx} ---")
            f_out.write(f"--- Function {func_idx} ---\n")
            d = dims[func_idx]
            
            # Dynamic Exploration/Exploitation Assignment (Week 2)
            if func_idx in [1, 2, 4, 5, 6]:
                # High exploration: active subspace, noisy, multimodal landscapes
                kappa = 4.0
                xi = 0.1
            elif func_idx == 3:
                # Moderate decay: 3D drug discovery, clearer gradient but still sparse (16 pts)
                kappa = 2.5
                xi = 0.05
            elif func_idx in [7, 8]:
                # Stay exploratory: 6D/8D still massively under-sampled, avoid phantom optima
                kappa = 3.5
                xi = 0.1
            
            # Load historical data
            f_dir = os.path.join(base_dir, f"function_{func_idx}")
            X = np.load(os.path.join(f_dir, "initial_inputs.npy"))
            y = np.load(os.path.join(f_dir, "initial_outputs.npy"))
            
            # 2. Consolidate Data
            new_x = np.array(new_inputs_list[func_idx - 1])
            new_y = new_outputs_list[func_idx - 1]
            
            X = np.vstack((X, np.atleast_2d(new_x)))
            y = np.append(y, new_y)
            
            # 3. Directory Structure & Save
            func_next_week_dir = os.path.join(next_week_dir, f"function_{func_idx}")
            os.makedirs(func_next_week_dir, exist_ok=True)
            np.save(os.path.join(func_next_week_dir, "initial_inputs.npy"), X)
            np.save(os.path.join(func_next_week_dir, "initial_outputs.npy"), y)
            
            # Trust region bounds (Functions 7 & 8) — adaptive width
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

            # Kernel definition based on function characteristics
            if func_idx == 2:
                kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 10.0))
            elif func_idx == 4:
                kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=1.5)
            else:
                kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)

            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True, n_restarts_optimizer=30)
            gp.fit(X, y)

            # Function 1: SVC active subspace — relative threshold (top 20th percentile)
            svc_model = None
            if func_idx == 1:
                threshold = np.percentile(y, 80)
                labels = (y > threshold).astype(int)
                if np.sum(labels) > 0 and np.sum(labels) < len(labels):
                    svc_model = SVC(kernel='rbf', probability=True)
                    svc_model.fit(X, labels)

            # Latin Hypercube Sampling — single n=10000 draw with fixed seed
            sampler = qmc.LatinHypercube(d=d, seed=42)
            X_sample = sampler.random(n=10000)

            # Interior-biased LHS for Function 3 only (Function 5 boundary is genuine signal)
            if func_idx in [3]:
                lhs_lower = np.array([0.05] * d)
                lhs_upper = np.array([0.95] * d)
            else:
                lhs_lower = np.array([b[0] for b in base_bounds])
                lhs_upper = np.array([b[1] for b in base_bounds])
            X_sample = qmc.scale(X_sample, lhs_lower, lhs_upper)

            # Apply SVC filter for func_idx == 1
            if svc_model is not None:
                preds = svc_model.predict(X_sample)
                if np.sum(preds == 1) > 0:
                    X_sample = X_sample[preds == 1]

            # Evaluate Acquisition on LHS samples
            if func_idx in [3, 5]:
                f_best   = np.max(y)
                acq_vals = ei(X_sample, gp, f_best, xi)
            else:
                acq_vals = ucb(X_sample, gp, kappa)

            # Multi-start L-BFGS-B from top 50 seeds
            full_bounds = [(1e-6, 0.999999) for _ in range(d)]
            n_seeds     = min(50, len(X_sample))
            top_X       = X_sample[np.argsort(acq_vals)[-n_seeds:]]

            best_x   = None
            best_val = -np.inf

            for x0 in top_X:
                if func_idx in [3, 5]:
                    res = minimize(obj_func_ei, x0, args=(gp, f_best, xi), bounds=full_bounds, method="L-BFGS-B")
                else:
                    res = minimize(obj_func_ucb, x0, args=(gp, kappa), bounds=base_bounds, method="L-BFGS-B")

                if -res.fun > best_val:
                    if svc_model is not None:
                        if svc_model.predict(res.x.reshape(1, -1))[0] == 0:
                            continue
                    best_val = -res.fun
                    best_x   = res.x

            if best_x is None:
                best_x = top_X[-1]

            # Clamp strictly to [0.000001, 0.999999]
            best_x = np.clip(best_x, 1e-6, 0.999999)

            # Validation Metrics
            mean_val, std_val = gp.predict(best_x.reshape(1, -1), return_std=True)
            mean_val = mean_val[0]
            std_val  = std_val[0]
            if func_idx in [3, 5]:
                acq_score = ei(best_x, gp, f_best, xi)[0]
            else:
                acq_score = mean_val + kappa * std_val
            
            metrics_str = f"Metrics -> GP Mean: {mean_val:.6f}, GP Std: {std_val:.6f}, Acq Score: {acq_score:.6f}"
            coord_str = "-".join([f"{x:.6f}" for x in best_x])
            
            print(metrics_str)
            print(f"Coordinate String: {coord_str}\n")
            f_out.write(metrics_str + "\n")
            f_out.write(f"Coordinate String: {coord_str}\n\n")

            # --- Plotting 2D Slices ---
            n_grid = 50
            x1 = np.linspace(0, 1, n_grid)
            x2 = np.linspace(0, 1, n_grid)
            X1, X2 = np.meshgrid(x1, x2)
            grid_pts = np.vstack((X1.flatten(), X2.flatten())).T
            
            grid_d = np.zeros((n_grid * n_grid, d))
            for i in range(d):
                if i == 0:
                    grid_d[:, i] = grid_pts[:, 0]
                elif i == 1:
                    grid_d[:, i] = grid_pts[:, 1]
                else:
                    grid_d[:, i] = best_x[i]
                    
            if func_idx in [3, 5]:
                f_best = np.max(y)
                acq_grid = ei(grid_d, gp, f_best, xi).reshape(n_grid, n_grid)
                acq_label = 'EI Acquisition Score'
            else:
                acq_grid = ucb(grid_d, gp, kappa).reshape(n_grid, n_grid)
                acq_label = 'UCB Acquisition Score'
            
            plt.figure(figsize=(8, 6))
            contour = plt.contourf(X1, X2, acq_grid, levels=50, cmap='viridis')
            plt.colorbar(contour, label=acq_label)
            
            if d == 2:
                plt.scatter(X[:, 0], X[:, 1], c='red', marker='x', label='Historical Data')
                # highlight the newest point
                plt.scatter(X[-1, 0], X[-1, 1], c='cyan', marker='o', edgecolors='black', label='New Point (W1)')
            else:
                plt.scatter(X[:, 0], X[:, 1], c='red', marker='x', alpha=0.5, label='Historical Data (Proj Dims 1&2)')
                plt.scatter(X[-1, 0], X[-1, 1], c='cyan', marker='o', edgecolors='black', label='New Point (W1)')
                
            plt.scatter(best_x[0], best_x[1], color='gold', s=200, edgecolors='black', marker='*', label='Suggested Point (W2)')
            
            plt.title(f"Function {func_idx} - Acq Surface (Slice Dims 1 & 2)")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.legend(loc='upper right', fontsize='small')
            
            plot_file = os.path.join(output_dir, f"plot_func{func_idx}_week2.png")
            plt.savefig(plot_file, dpi=150)
            plt.close()

if __name__ == '__main__':
    run_bbo()
