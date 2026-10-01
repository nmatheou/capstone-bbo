# Reproducibility Guide

To reproduce the weekly optimisation outputs, make sure the repository structure remains unchanged and follow the steps below:

1. Environment Setup  
Install the core scientific computing stack:  
`bash  
pip install numpy scipy scikit-learn matplotlib  
`

2. Running the Optimiser  
The optimiser must be executed from the repository root directory. It requires the --week argument in order to dynamically identify the correct input data and output directories.

`bash  
python src/bbo_optimize.py --week 5  
`  
*Note: --week 5 will automatically look for data in data/week_4_returns/ and output suggestions to  
results/week_5/suggestions.txt.*

3. Modifying Parameters  
Hyperparameters ($\kappa$, $\xi$, and random seeds) are explicitly separated from the Python source code. Update configs/hyperparameters.json before running the optimiser to modify exploration thresholds.
