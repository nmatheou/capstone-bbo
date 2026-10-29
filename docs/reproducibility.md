# Reproducibility Guide & Required Project Structure

This guide specifies the complete repository directory hierarchy, runtime configuration mechanics, execution prerequisites, and command instructions required to reproduce all optimisation runs, diagnostic dashboards, and final submissions without errors.

---

## 1. Required Directory & File Structure

The pipeline relies on a clean, modular structure. Below is the full directory tree expected by `src/bbo_optimize.py` and diagnostic tools:

```text
capstone-bbo/
├── configs/
│   ├── hyperparameters.json        # Base acquisition thresholds (kappa, xi) and LHS random seeds
│   └── runtime_config.json         # [Auto-generated on first run] Local machine output path (git-ignored)
├── data/
│   ├── accumulated/                # Authoritative cumulative datasets across all rounds
│   │   ├── function_1/             # initial_inputs.npy, initial_inputs.csv, initial_outputs.npy, ...
│   │   ├── function_2/
│   │   ├── function_3/
│   │   ├── function_4/
│   │   ├── function_5/
│   │   ├── function_6/
│   │   ├── function_7/
│   │   └── function_8/
│   ├── week_1_returns/             # Raw weekly oracle query returns: inputs.txt, outputs.txt
│   ├── ...
│   └── week_13_returns/
├── docs/                           # Governance, technical, and architectural documentation
│   ├── architecture.md             # End-to-end software and surrogate model architecture
│   ├── challenges_and_improvements.md
│   ├── datasheet.md                # Module 21 Dataset Datasheet
│   ├── diagnostics_and_overrides.md
│   ├── model_card.md               # Module 25 Model Card
│   └── reproducibility.md          # This reproducibility guide and filesystem specification
├── logs/
│   └── run_logs.txt                # Append-only execution audit log with timestamps and run metadata
├── results/
│   ├── week_1/ ... week_13/
│   │   ├── suggestions.txt         # Diagnostic log of acquisition scores, GP/NN metrics, and coordinates
│   │   ├── plots/                  # 2D acquisition landscape contour plots (plot_func<1..8>.png)
│   │   └── progress/               # Trajectory, normalized metrics, and length-scale dashboards
├── src/                            # Production optimization engine and diagnostic tooling
│   ├── bbo_optimize.py             # Main CLI optimization engine (GP, NN surrogate, UCB/EI, refinement)
│   ├── paths.py                    # Path resolution, first-run interactive setup, and validation checks
│   ├── test_paths.py               # Automated sanity checks for paths and data integrity
│   ├── progress_tracker.py         # 13-week performance trajectory generator
│   ├── plot_normalized_metrics.py  # Normalized GP metrics dashboard generator
│   ├── length_scale_tracker.py     # ARD length-scale dynamics and stability tracker
│   ├── dim_sensitivity_sweep.py    # 1D slice sensitivity analyzer
│   └── pca_analysis.py             # Principal component variance analysis
├── submissions/
│   ├── week_1/ ... week_13/        # Locked weekly submission coordinates formatted for the Oracle
│   │   └── suggestions.txt         # Final, locked coordinate strings
├── bbo_capstone_presentation.ipynb # Interactive presentation notebook with embedded visuals
└── README.md                       # Master project overview, methodology, and results index
```

---

## 2. Dynamic Path Resolution & First-Run Setup

To guarantee that the codebase runs seamlessly on any machine (whether cloned by a student, marker, or CI/CD environment) without crashing on hardcoded personal paths, the codebase employs a self-configuring path resolver in [`src/paths.py`](../src/paths.py):

### First Run Behavior
When any script in `src/` is executed for the first time:
1. **Interactive Terminal (`sys.stdin.isatty()` is `True`):** The script detects that no runtime configuration exists, displays the detected repository root, and asks the user:
   ```text
   ========================================================================
   [BBO Setup - Initial Configuration]
   No valid output directory configuration was found.
   Detected repository root:
     /path/to/capstone-bbo
   ========================================================================
   Press [Enter] to use repository root, or enter custom path: 
   ```
   * Pressing **[Enter]** immediately sets the output directory to the repository root.
   * Typing a custom path and confirming sets the output directory to that custom location (creating the folder if requested).
   * The selection is saved to `configs/runtime_config.json`.
2. **Non-Interactive Execution (Jupyter Notebook, CI/CD, piping):** The script automatically defaults to the detected repository root, initializes `configs/runtime_config.json`, and proceeds without blocking.

### Subsequent Runs & Validation
* On every subsequent run, the script loads `configs/runtime_config.json` and **checks that the path exists on disk before executing**.
* If the folder has been moved or deleted, the script catches the issue, displays a descriptive `FileNotFoundError` with remediation guidance, and prompts for re-configuration.
* **Command-Line Override:** You can override the output directory at any time using the `--output-dir` flag:
  ```bash
  python src/bbo_optimize.py --week 13 --output-dir /custom/path
  ```
* **Environment Variable Override:** Setting the `BBO_OUTPUT_DIR` environment variable takes precedence over `configs/runtime_config.json`.

---

## 3. Execution Prerequisites Checklist

Before executing `python src/bbo_optimize.py --week <N>`, ensure the following files exist:

| Required Input File | Location | Description |
| :--- | :--- | :--- |
| **Accumulated Inputs** | `data/accumulated/function_<1..8>/initial_inputs.npy` | Feature arrays $\mathbf{X}$ for all 8 functions |
| **Accumulated Outputs** | `data/accumulated/function_<1..8>/initial_outputs.npy` | Scalar scores $\mathbf{y}$ for all 8 functions |
| **Hyperparameter Config** | `configs/hyperparameters.json` | Exploration thresholds ($\kappa, \xi$) and LHS seed |
| **Previous Oracle Returns** | `data/week_<N-1>_returns/inputs.txt` & `outputs.txt` | Prior week's returned points (required for $N \ge 2$) |

*Note on Output Directories:* The script automatically creates `results/week_<N>/`, `results/week_<N>/plots/`, `logs/`, and `submissions/week_<N>/` if they do not already exist.

---

## 4. Step-by-Step Reproduction Instructions

### Step 1: Environment Setup
Ensure Python 3.10+ is installed along with the required scientific computing stack and PyTorch:

```bash
pip install numpy scipy scikit-learn matplotlib torch
```

### Step 2: Validate Paths & Data Integrity
Run the built-in path and dataset validator from the repository root:

```bash
python src/test_paths.py
```
*Expected output:* `test_paths: all checks passed (including first-run & validation logic)`

### Step 3: Test Run (Dry-Run Mode)
To simulate fitting the models and calculating suggestions without writing to disk:

```bash
python src/bbo_optimize.py --week 13 --dry-run
```

### Step 4: Full Optimization Run
To execute the complete optimization pipeline for Week 13:

```bash
python src/bbo_optimize.py --week 13
```
This command will:
1. Read the previous week's oracle return files (`data/week_12_returns/inputs.txt` and `outputs.txt`).
2. Verify observations against `data/accumulated/function_<1..8>/`.
3. Fit the seeded Gaussian Processes (`random_state=42`) and PyTorch 64-64 ReLU neural network (`torch.manual_seed(42)`).
4. Execute `compute_final_refinement` for all 8 functions.
5. Save diagnostic acquisition landscapes to `results/week_13/plots/`.
6. Write full diagnostic logs to `results/week_13/suggestions.txt` and append to `logs/run_logs.txt`.
7. Write the locked, exact 8 submission coordinates to `submissions/week_13/suggestions.txt`.

### Step 5: Regenerate Visual Dashboards
To regenerate the 13-week diagnostic tracking figures:

```bash
# 1. 13-Week Optimization Trajectory Dashboard
python src/progress_tracker.py --week 13

# 2. Normalized GP Performance Metrics Dashboard
python src/plot_normalized_metrics.py --week 13

# 3. ARD Kernel Length-Scale Dynamics Dashboard
python src/length_scale_tracker.py --week 13

# 4. 1D Dimensional Slice Sensitivity Sweep
python src/dim_sensitivity_sweep.py --week 13
```
All generated figures are saved to `results/week_13/progress/`.

### Step 6: Interactive Presentation Notebook
Launch Jupyter to review and run the complete interactive narrative:

```bash
jupyter notebook bbo_capstone_presentation.ipynb
```
The notebook executes end-to-end and renders all 13-week progression tables and visual dashboards natively.
