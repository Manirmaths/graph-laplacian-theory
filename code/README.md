# Code for "A Graph-Laplacian Theory of Forgetting and Local Memory Capacity in Adaptive Resistor Networks"

These scripts produce the numbers and figures in the manuscript.

```
pip install -r requirements.txt
bash run_all.sh
```

Run from inside this folder. The full run takes about 15 seconds. Results are saved as `.npy` and `.json` files, and figures are written to `figures/`. The saved results are included, so the figure scripts can also be run on their own.

| Script | Contents |
|---|---|
| `01_network.py` | 5x5 grid, reduced Laplacian and analytic Jacobian; loaded by the other scripts |
| `02_reference_network.py` | Checks on the untrained reference network |
| `03_train_task_A.py` | Trains task A to exact fit |
| `04_experiments_main.py` | Range of the quadratic law, anchoring sweeps, grounding table, current vs sensitivity, rank growth |
| `05_overlap_and_projection.py` | Gamma vs forgetting over 55 scalar task pairs; plain vs projected training |
| `06_gamma_scan.py` | Gamma for candidate task pairs |
| `07_figures.py`, `08_figures_trained.py` | Figures for the reference and trained networks |
| `09_verification_checks.py` | Manufactured anchoring test, quartic coefficient, complex-step Jacobian check, rank of multi-output tasks, Gamma outlier |
| `10_figures_verification.py` | Jacobian check, quadratic law, pathway maps, Gamma and current-vs-sensitivity figures |
