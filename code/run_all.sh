#!/usr/bin/env bash
# Reproduces every number and figure in the manuscript.
# Run from inside this directory:  bash run_all.sh
# Requires: see requirements.txt (pip install -r requirements.txt)
set -e
mkdir -p figures
echo "== 02: reference network checks =="
python3 02_reference_network.py
echo; echo "== 04: main experiments (local reach, anchoring, rank, pathway, quartic) =="
python3 04_experiments_main.py
echo; echo "== 05: task overlap and null-space projection =="
python3 05_overlap_and_projection.py
echo; echo "== 06: Gamma scan over candidate task pairs =="
python3 06_gamma_scan.py
echo; echo "== 07/08: figures =="
python3 07_figures.py
python3 08_figures_trained.py
echo; echo "== 09/10: verification checks and figures =="
python3 09_verification_checks.py
python3 10_figures_verification.py
echo; echo "Done. Figures written to ./figures"
