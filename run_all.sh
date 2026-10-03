#!/bin/sh
# Reproduces every number and figure in Section X-J of the paper (single core: about 55 min).
set -e
cd "$(dirname "$0")/experiments"
python3 verify_model.py            # V1-V6       -> results/verification.json
python3 case33.py ica              # Stage 3/5   -> results/case33_ica.pkl
python3 case33.py coalition        # Stage 6     -> results/case33_coalition.pkl
python3 case33.py ablation         # V6 normalization comparison
python3 case33.py cc               # chance-constrained variant
python3 case33.py trip             # Scenario 7 (G1 trip at minute 30)
python3 case33.py central 600      # centralized multi-island MILP, 600 s limit (as in the paper)
python3 case33.py report           # -> results/case33_summary.json
python3 sensitivity.py df          # Fig. 13: frequency-limit sweep
python3 sensitivity.py scale       # Fig. 14: solve time vs island size
python3 sensitivity.py eps         # Fig. 12: risk-level sweep with Monte Carlo
python3 make_figures.py            # Figs. 2, 4, 15
python3 more_figures.py            # Figs. 1, 3, 5-14
