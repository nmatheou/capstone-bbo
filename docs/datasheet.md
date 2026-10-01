# Datasheet for Black-Box Optimization (BBO) Datasets

This document provides a description of the sequential datasets collected as part of the Imperial College Capstone project on Black-Box Optimization.

## 1. Motivation

**Purpose of Dataset:**
This dataset was constructed in an iterative manner to tackle a tightly constrained Black-Box Optimization (BBO) problem. As the governing analytical expressions of the systems were fully hidden (black-box), the only viable approach to characterising the structure of the search spaces was through empirical data generation.

**Supporting Task:**
The dataset is used to train surrogate models, including Gaussian Processes and shallow neural networks, in order to optimise a scalar reward over eight different functions. These functions simulate real-world optimisation tasks such as radioactive source localisation, synthetic environments, drug discovery, and hyperparameter optimisation.

## 2. Composition

**Contents of Dataset:**
The dataset consists of eight separate subsets (Functions 1 to 8). Each subset includes pairs of continuous input coordinates `$X$` within the range `[0,1]^d`, where the dimensionality `$d$` varies from 2 to 8, along with a corresponding deterministic scalar output `$Y$`.

**Size and Format:**
*   **Size:** The dataset is highly sparse. By Week 10, each function contains approximately 29-30 data points, consisting of an initial blind sampling batch plus 10 sequential query points.
*   **Format:** Raw entries are stored as `.txt` strings from weekly submissions and later consolidated into NumPy `.npy` files (`initial_inputs.npy` and `initial_outputs.npy`).

**Data Gaps:**
Significant gaps exist due to the inherent sparsity of sampling in high-dimensional spaces. For example, in 8D Function 8, 30 points represent an extremely small fraction of the overall domain. The dataset also reflects intentional sampling bias, prioritising exploitation of known high-value regions while neglecting low-information areas. In Function 1 (Sparse Needle), most values are zero, with only a small concentrated region of positive outputs.

## 3. Collection Process

**Query Process:**
Data was generated sequentially over a 10-week period, with one query per function per week, using active learning approaches.

**Query Strategy:**
The strategy evolved from pure surrogate-based optimisation to a hybrid Human-in-the-Loop method. Initially, a Gaussian Process model with a Matérn kernel and Expected Improvement acquisition function guided sampling. Over time, limitations became apparent, including oversmoothing of discontinuities, epistemic artefacts near boundaries, and ARD length-scale collapse that reduced effective dimensionality. As a result, shallow neural networks and human input were introduced to encourage exploration, correct covariance blind spots, and perform boundary-constrained adjustments.

**Collection Timeframe:**
All data was collected during a 10-week sequential submission period for the Capstone project.

## 4. Pre-processing and Uses

**Transformations Applied:**
Before model training, target values `$Y$` were standardised to zero mean and unit variance to improve Gaussian Process kernel stability and neural network loss behaviour.

**Data Pruning:**
For Function 1, additional preprocessing was required. Because the dataset is dominated by near-zero values, neural network loss functions were unable to detect the localized peak structure. A targeted pruning step was therefore applied, removing global samples and restricting training to a small bounding region around the anomaly to better capture local gradients.

**Use Case:**
The dataset is intended for local surrogate modelling tasks aimed at identifying the global maximum or best achievable local maximum under strict query constraints.

**Limitations:**
The dataset is not suitable for learning full global structure across the eight functions. Due to strong sampling bias toward already explored regions, global models trained on this data tend to overfit to sparse clusters and produce unreliable predictions in unobserved areas.

## 5. Distribution and Maintenance

**Data Storage:**
The dataset is stored locally within the project repository under the directories `data/accumulated/` and `results/`.

**Terms of Use:**
Use of the dataset is restricted to academic evaluation in accordance with Imperial College Capstone project guidelines.

**Maintenance:**
The dataset is maintained by the student researcher. No further updates or additions will be made after the completion of Week 12.
