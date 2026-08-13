import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import warnings
warnings.filterwarnings("ignore")

def compare_models():
    data_dir = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/antigravity/next_week_data/function_1"
    
    try:
        X = np.load(os.path.join(data_dir, "initial_inputs.npy"))
        y = np.load(os.path.join(data_dir, "initial_outputs.npy"))
    except FileNotFoundError:
        print("Data for Function 1 not found. Make sure the path is correct.")
        return

    print(f"Loaded {X.shape[0]} data points for Function 1.\n")

    # --- 1. Define Models ---
    # GP Model
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 0.5), nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True, n_restarts_optimizer=30, random_state=42)

    # MLP Model
    mlp_base = MLPRegressor(hidden_layer_sizes=(4,), activation='tanh', alpha=0.001, solver='lbfgs', max_iter=2000, random_state=42)
    mlp = TransformedTargetRegressor(regressor=mlp_base, transformer=StandardScaler())

    models = {'Gaussian Process (Matern 2.5)': gp, 'Neural Net (MLP 4-tanh)': mlp}
    
    # --- 2. Leave-One-Out Cross-Validation (LOOCV) ---
    print("--- Leave-One-Out Cross-Validation (LOOCV) ---")
    loo = LeaveOneOut()
    
    for name, model in models.items():
        y_true = []
        y_pred = []
        
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            y_true.append(y_test[0])
            y_pred.append(pred[0])
            
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R^2: {r2:.4f}\n")

    # --- 3. Visual Surface Comparison ---
    print("Generating Surface Comparison Plot...")
    # Train on all data for the plot
    gp.fit(X, y)
    mlp.fit(X, y)
    
    n_grid = 50
    x1_g = np.linspace(0, 1, n_grid)
    x2_g = np.linspace(0, 1, n_grid)
    X1, X2 = np.meshgrid(x1_g, x2_g)
    grid_pts = np.vstack((X1.flatten(), X2.flatten())).T

    gp_preds = gp.predict(grid_pts).reshape(n_grid, n_grid)
    mlp_preds = mlp.predict(grid_pts).reshape(n_grid, n_grid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # GP Plot
    c1 = axes[0].contourf(X1, X2, gp_preds, levels=50, cmap='viridis')
    axes[0].scatter(X[:, 0], X[:, 1], c='red', marker='x', label='Data')
    axes[0].set_title('Gaussian Process Surface')
    fig.colorbar(c1, ax=axes[0])
    
    # MLP Plot
    c2 = axes[1].contourf(X1, X2, mlp_preds, levels=50, cmap='viridis')
    axes[1].scatter(X[:, 0], X[:, 1], c='red', marker='x', label='Data')
    axes[1].set_title('Neural Network Surface')
    fig.colorbar(c2, ax=axes[1])
    
    plot_path = "C:/Users/NtecD/OneDrive/AI/Imperial/Capstone/antigravity/model_comparison_F1.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")

if __name__ == '__main__':
    compare_models()
