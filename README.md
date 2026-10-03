# Adaptive MAS service restoration — island model, verification, and case study

Companion code for *Adaptive Multi-Agent Service Restoration via Dynamic Islanding and
Uncertainty-Aware DER Coordination* (V7). It implements the island-level optimization model of
Sections III–IV and VII-D(ii) of the paper, the post-solve checks of Sections III-G and VI-C, and
reproduces every number in Section X-J.

## What is implemented

| Paper element | File |
|---|---|
| Island MILP (1)–(17), objective (19)–(22), chance constraint (27); single-island (ICA) and island-indexed multi-island (B1) forms | `restoration/model.py` |
| Graph check, identity (17), dwell check, AC validation with re-dispatch/shedding policy | `restoration/checks.py` |
| Sections, nearest-reference candidates, BFS shells, adequacy screen (18) | `restoration/candidates.py` |
| IEEE 33-bus feeder (Baran–Wu data) with declared DER/BESS/EV overlay; test networks | `restoration/data.py` |
| Verification suite V1–V6 | `experiments/verify_model.py` |
| Case study: ICA, coalition rule (25), centralized, normalization ablation, chance-constrained Monte Carlo, DG trip (Scenario 7) | `experiments/case33.py` |
| Sensitivity sweeps (frequency limit, risk level, solve time vs size) | `experiments/sensitivity.py` |
| Figures 1–15 of the paper | `experiments/make_figures.py`, `experiments/more_figures.py` |

## What is NOT implemented (and is not claimed in the paper)

* The agent protocol (election, leases, heartbeats, fallback (28)) and the communication model;
  the case study calls the island solver directly.
* The scenario-based and robust uncertainty treatments (only deterministic and (27)).
* Baselines B2 (re-implementation of the 2015 MAS), B3 (GA), B4 (ADMM).
* The other test feeders (69-, 119-, 123-bus), the sweeps of Section X-B/X-D, and time-domain
  frequency simulation.
* Cold-load pickup dynamics (load forecast multiplier is 1.0).

The case study is therefore a verification of the model on one feeder with an assumed resource
overlay, **not** the comparative evaluation of Section X.

## Running

```
pip install -r requirements.txt
./run_all.sh
```

Solver: HiGHS through PuLP 2.x (PuLP 4 changed its variable API and is not supported). Results are
deterministic for a given HiGHS version; solve times depend on the machine. The paper's numbers
were produced on a single CPU core.

## Units and conventions

All model quantities are per unit on `Network.S_base` (1000 kVA for the 33-bus case);
`Resource` data are in kW/kvar/kWh. Interval `t` covers `[(t-1)Δt, tΔt)`; `SOC[b,t]` is the state
at the beginning of interval `t`. Edge orientation is `(i, j)` as listed in the data.
