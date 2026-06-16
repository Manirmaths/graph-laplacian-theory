# Graph-Laplacian theory verification

This package generates the numerical diagnostics for the manuscript
**A Graph-Laplacian Theory of Forgetting and Memory Capacity in Adaptive Resistor Networks**.

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python -m experiments.run_theory_verification
```

Results are written to:

```text
results/theory_verification/
├── summary.json
├── data/
└── figures/
```

Every figure is exported as both PDF and PNG. CSV files contain the plotted
values and fitted asymptotic diagnostics.

## Important theoretical correction

For an exactly fitted old task, if `J_A @ delta_g = 0`, then the residual change
is second order and the squared loss is fourth order:

```text
F_A(delta_g) = O(||delta_g||^4).
```

This is sharper than the earlier `O(||delta_g||^3)` Taylor bound and should be
used in the manuscript theorem and numerical section.
