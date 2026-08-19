# Reproducibility Guide

To reproduce the weekly optimization outputs, ensure the repository structure is intact and follow these steps:

## 1. Environment Setup
Install the core mathematical stack:
`ash
pip install numpy scipy scikit-learn matplotlib
`

## 2. Running the Optimizer
The optimizer must be run from the root directory of the repository. It requires the --week argument to dynamically locate the correct input data and output folders.

`ash
python src/bbo_optimize.py --week 5
`
*Note: --week 5 will automatically look for data in data/week_4_returns/ and output suggestions to esults/week_5/suggestions.txt.*

## 3. Modifying Parameters
Hyperparameters ($\kappa$, $\xi$, and random seeds) are explicitly decoupled from the python source code. Modify configs/hyperparameters.json before running the optimizer to adjust exploration thresholds.
