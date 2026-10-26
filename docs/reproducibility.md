# Reproducibility Guide

To reproduce the weekly optimisation outputs, including the final Week 13 submission, ensure the repository structure remains unchanged and follow the steps below:

## 1. Environment Setup
Install the required scientific computing stack and PyTorch (used for the secondary 64-64 ReLU Neural Network surrogate):

```bash
pip install numpy scipy scikit-learn matplotlib torch
```

## 2. Running the Optimiser
Execute the optimiser from the repository root directory. Pass the `--week` argument to dynamically resolve the previous week's oracle returns (`data/week_<N-1>_returns/`) and write outputs to `results/week_<N>/` and (for `--week 13`) `submissions/week_<N>/suggestions.txt`:

```bash
python src/bbo_optimize.py --week 13
```

*Note:*
* Running `python src/bbo_optimize.py --week 13` reads `data/week_12_returns/inputs.txt` and `outputs.txt`, verifies that all Week 12 observations are present in `data/accumulated/function_<1..8>/`, fits the seeded Gaussian Process (`random_state=42`) and PyTorch 64-64 ReLU Neural Network (`torch.manual_seed(42)`), executes `compute_final_refinement` for all 8 functions, and writes the exact 8 locked submission coordinates to `submissions/week_13/suggestions.txt` and `results/week_13/suggestions.txt`.
* To regenerate the Week 12 progress and length-scale dashboards:
  ```bash
  python src/progress_tracker.py --week 12
  python src/plot_normalized_metrics.py --week 12
  python src/length_scale_tracker.py --week 12
  ```

## 3. Modifying Parameters
Base acquisition hyperparameters ($\kappa$, $\xi$, and Latin Hypercube random seeds) are separated from the Python source code in `configs/hyperparameters.json`. Update `configs/hyperparameters.json` before running the optimiser to modify base exploration thresholds.
