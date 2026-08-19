# Challenges & Future Improvements

## Known Limitations
1. **Length-Scale Saturation:** In high-dimensional spaces with sparse data (e.g., F8 with 44 points in 8D), the Matern kernel frequently assigns maximum length scales to critical dimensions, effectively ignoring them.
2. **Boundary Flatness:** GPs fit smooth, continuous curves. They struggle significantly to model step-functions or perfectly flat slopes that terminate at a physical boundary.

## Mitigation Strategies Employed
- Aggressive bounds restriction (via Latin Hypercube bounding in F3).
- Manual override of the surrogate model to force boundaries.

## Future Work
- **Automated Acquisition Switching:** Implementing a script that automatically flips a function from UCB to EI once the CV drops below a specific threshold.
- **Multi-fidelity BO:** Integrating lower-fidelity/cheaper approximations of the Oracle if they become available.
