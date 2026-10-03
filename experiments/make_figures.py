"""Figures for Section X-J (saved to results/figs/)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from restoration.model import secant_breakpoints  # noqa: E402
import case33 as C  # noqa: E402

FIG = os.path.join(C.OUT, "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3})
TIER_COL = {1: "#b2182b", 2: "#ef8a62", 3: "#67a9cf"}


def served_by_tier(results):
    ev = C.evaluate(results)
    s = ev["served"]
    return {tier: np.array([sum(s.get((i, t), 0) for i in C.NORM["aff"] if C.NET.loads[i].tier == tier) * 1000
                            for t in range(1, C.H + 1)]) for tier in (1, 2, 3)}


def fig_restoration():
    cfgs = [("ICA candidates", C.load("ica")), ("ICA + coalition (25)", C.load("coalition")["results"])]
    if os.path.exists(os.path.join(C.OUT, "case33_central.pkl")):
        cfgs.append(("Centralized MILP", C.load("central")))
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.0 * len(cfgs), 2.4), sharey=True)
    tmin = np.arange(C.H) * 5
    for ax, (name, res) in zip(np.atleast_1d(axes), cfgs):
        st = served_by_tier(res)
        ax.stackplot(tmin, st[1], st[2], st[3], colors=[TIER_COL[1], TIER_COL[2], TIER_COL[3]],
                     labels=["Tier 1", "Tier 2", "Tier 3"], step="post", alpha=0.9)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("time after isolation (min)")
    np.atleast_1d(axes)[0].set_ylabel("served load (kW)")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_restoration.png"), dpi=200)


def fig_trip():
    if not os.path.exists(os.path.join(C.OUT, "case33_trip.pkl")):
        return
    tr = C.load("trip")
    bef = served_by_tier(tr["before"])
    aft = served_by_tier(tr["after"])
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    t_b = np.arange(6) * 5
    t_a = 30 + np.arange(C.H) * 5
    for tier in (1, 2, 3):
        ax.step(np.r_[t_b, t_a], np.r_[bef[tier][:6], aft[tier]], where="post", color=TIER_COL[tier],
                label=f"Tier {tier}")
    ax.axvline(30, color="k", ls="--", lw=0.8)
    ax.text(31, ax.get_ylim()[1] * 0.92, "G1 trips", fontsize=7)
    ax.set_xlabel("time after isolation (min)")
    ax.set_ylabel("served load (kW)")
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_trip.png"), dpi=200)


def fig_secant():
    xs = np.linspace(1e-3, 1, 500)
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for sp, ls in (("uniform", "-"), ("quadratic", "--")):
        bp = secant_breakpoints(1.0, 8, sp)
        ch = np.max(np.stack([(a + b) * xs - a * b for a, b in zip(bp[:-1], bp[1:])]), axis=0)
        ax.plot(xs, 100 * (ch - xs ** 2) / xs ** 2, ls, label=f"{sp} breakpoints")
    ax.set_yscale("log")
    ax.set_xlabel("branch flow / rating")
    ax.set_ylabel("loss over-estimate (%)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_secant.png"), dpi=200)


if __name__ == "__main__":
    fig_secant(); fig_restoration(); fig_trip()
    print("figures in", FIG)
