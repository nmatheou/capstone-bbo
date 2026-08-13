import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.svm import SVC
from scipy.stats import qmc, norm
from scipy.optimize import minimize
import warnings

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
    results_file = os.path.join(output_dir, "results_week1.txt")
    dims = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}

    with open(results_file, 'w') as f_out:
        for func_idx in range(1, 9):
            print(f"--- Function {func_idx} ---")
            f_out.write(f"--- Function {func_idx} ---\n")
            d = dims[func_idx]
            
            # Dynamic Kappa Assignment
            if func_idx in [4, 6]:
                kappa = 3.0
            else:
                kappa = 4.0
            
            # Load data
            f_dir = os.path.join(base_dir, f"function_{func_idx}")
            X = np.load(os.path.join(f_dir, "initial_inputs.npy"))
            y = np.load(os.path.join(f_dir, "initial_outputs.npy"))
            
            # Trust region bounds (Functions 7 & 8)
            base_bounds = [(1e-6, 0.999999) for _ in range(d)]
            if func_idx in [7, 8]:
                best_idx = np.argmax(y)
                x_best = X[best_idx]
                tr_width = 0.2
                base_bounds = [(max(1e-6, x_best[i] - tr_width/2), min(0.999999, x_best[i] + tr_width/2)) for i in range(d)]

            iter_best_val = -np.inf
            iter_best_x = None
            iter_best_gp = None

            for _ in range(5):
                # Kernel definition based on function characteristics
                if func_idx == 2:
                    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 10.0))
                elif func_idx == 4:
                    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=1.5)
                else:
                    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                    
                gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True, n_restarts_optimizer=30)
                gp.fit(X, y)
                
                # Function 1: SVC active subspace
                svc_model = None
                if func_idx == 1:
                    svc_model = SVC(kernel='rbf', probability=True)
                    labels = (y > 1e-5).astype(int)
                    if np.sum(labels) > 0 and np.sum(labels) < len(labels):
                        svc_model.fit(X, labels)
                    else:
                        svc_model = None

                # Latin Hypercube Sampling (LHS)
                sampler = qmc.LatinHypercube(d=d)
                X_sample = sampler.random(n=5000)
                
                # Scale LHS samples to base_bounds
                lower_bounds = np.array([b[0] for b in base_bounds])
                upper_bounds = np.array([b[1] for b in base_bounds])
                X_sample = qmc.scale(X_sample, lower_bounds, upper_bounds)
                
                # Apply SVC filter for func_idx == 1
                if svc_model is not None:
                    preds = svc_model.predict(X_sample)
                    if np.sum(preds == 1) > 0:
                        X_sample = X_sample[preds == 1]
                
                # Evaluate Acq on LHS samples
                if func_idx in [3, 5]:
                    f_best = np.max(y)
                    xi = 0.1
                    acq_vals = ei(X_sample, gp, f_best, xi)
                else:
                    acq_vals = ucb(X_sample, gp, kappa)
                    
                # Multi-start L-BFGS-B from top 50 points
                n_seeds = min(50, len(X_sample))
                top_indices = np.argsort(acq_vals)[-n_seeds:]
                top_X = X_sample[top_indices]
                
                best_x = None
                best_val = -np.inf
                
                for x0 in top_X:
                    if func_idx in [3, 5]:
                        res = minimize(obj_func_ei, x0, args=(gp, f_best, xi), bounds=base_bounds, method="L-BFGS-B")
                    else:
                        res = minimize(obj_func_ucb, x0, args=(gp, kappa), bounds=base_bounds, method="L-BFGS-B")
                        
                    if -res.fun > best_val:
                        if svc_model is not None:
                            if svc_model.predict(res.x.reshape(1, -1))[0] == 0:
                                continue
                        best_val = -res.fun
                        best_x = res.x
                        
                if best_x is None:
                    best_x = top_X[-1]
                
                # Clamp strictly to requested range
                best_x = np.clip(best_x, 1e-6, 0.999999)
                
                # Output Validation Metrics for this iteration
                mean_val, std_val = gp.predict(best_x.reshape(1, -1), return_std=True)
                mean_val = mean_val[0]
                std_val = std_val[0]
                if func_idx in [3, 5]:
                    acq_score = ei(best_x, gp, f_best, xi)[0]
                else:
                    acq_score = mean_val + kappa * std_val

                if acq_score > iter_best_val:
                    iter_best_val = acq_score
                    iter_best_x = best_x
                    iter_best_gp = gp

            # Finalize best value across all 5 iterations
            best_x = iter_best_x
            acq_score = iter_best_val
            gp = iter_best_gp
            mean_val, std_val = gp.predict(best_x.reshape(1, -1), return_std=True)
            mean_val = mean_val[0]
            std_val = std_val[0]
            
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
                xi = 0.1
                acq_grid = ei(grid_d, gp, f_best, xi).reshape(n_grid, n_grid)
                acq_label = 'EI Acquisition Score'
            else:
                acq_grid = ucb(grid_d, gp, kappa).reshape(n_grid, n_grid)
                acq_label = 'UCB Acquisition Score'
            
            plt.figure(figsize=(8, 6))
            contour = plt.contourf(X1, X2, acq_grid, levels=50, cmap='viridis')
            plt.colorbar(contour, label=acq_label)
            
            if d == 2:
                plt.scatter(X[:, 0], X[:, 1], c='red', marker='x', label='Initial Data')
            else:
                plt.scatter(X[:, 0], X[:, 1], c='red', marker='x', alpha=0.5, label='Initial Data (Projected Dims 1&2)')
                
            plt.scatter(best_x[0], best_x[1], color='gold', s=200, edgecolors='black', marker='*', label='Suggested Point')
            
            plt.title(f"Function {func_idx} - Acq Surface (Slice Dims 1 & 2)")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.legend()
            
            plot_file = os.path.join(output_dir, f"plot_func{func_idx}.png")
            plt.savefig(plot_file, dpi=150)
            plt.close()

if __name__ == '__main__':
    run_bbo()
