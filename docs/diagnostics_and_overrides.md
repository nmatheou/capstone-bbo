# Diagnostics & Manual Overrides

Because Gaussian Processes perform poorly in high-dimensional spaces, a fully automated method is not adequate. This project therefore uses a hybrid approach.

## Algorithmic Diagnostics

*   **Coefficient of Variation (CV):** We monitor the standard deviation relative to the mean of past queries. A high CV suggests a highly irregular search space that requires exploration (UCB), while a low CV indicates a smoother space suitable for exploitation (Expected Improvement).
*   **Length-Scale Tracking:** When a GP length-scale reaches its maximum limit (e.g., `100.0`), that dimension is marked as "irrelevant".

## Manual Overrides

When the surrogate model underperforms, human intervention becomes necessary:

*   **Multimodal Behaviour (e.g., F1):** When the GP becomes stuck in "background radiation" (flat zero outputs), physical domain reasoning is applied. For F1, an anomaly was interpreted as a radiation shield, and offset coordinates were manually tested to identify the true source.
*   **Triangulation (e.g., F2, F6):** For dimensions with known peaks, the strongest historically performing coordinates are manually averaged (triangulated). In F6, information from a competing team was also incorporated into this triangulation.
*   **Boundary Locking (e.g., F8):** In high-dimensional cases (8D), the GP is unable to accurately model linear slopes. In these situations, the algorithm is manually overridden, forcing linear dimensions directly to their boundary values (`0.0` or `1.0`), while only the truly interior dimensions are left for triangulation.
