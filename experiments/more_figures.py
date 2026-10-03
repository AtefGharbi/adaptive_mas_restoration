"""Additional result figures for Section X-J (results/figs/fig03..fig14)."""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon, Circle
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import case33 as C  # noqa: E402
from restoration import checks  # noqa: E402

FIG = os.path.join(C.OUT, "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
NET, Sb, H = C.NET, C.NET.S_base, C.H
TMIN = np.arange(H) * 5
ISL_COL = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
TIER_COL = {1: "#b2182b", 2: "#ef8a62", 3: "#67a9cf"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ layout of the 33-bus feeder
POS = {i: (i - 1, 0.0) for i in range(1, 19)}
POS.update({19: (1, -1.3), 20: (2, -1.3), 21: (3, -1.3), 22: (4, -1.3)})
POS.update({23: (2, 1.3), 24: (3, 1.3), 25: (4, 1.3)})
POS.update({i: (i - 21, -2.6) for i in range(26, 34)})


def island_of(sol, t):
    lab = {}
    for (b, k, tt), v in sol["y"].items():
        if tt == t and v > 0.5:
            lab[b] = k
    return lab


def draw_network(ax, results, t, title):
    # global island id per result/k
    bus_isl, closed, refs = {}, set(), set()
    gid = 0
    for r in results:
        sol = r["sol"]
        G = checks.energized_graph(sol, sol["meta"], t)
        for comp in nx.connected_components(G):
            for b in comp:
                bus_isl[b] = gid
            gid += 1
        for e in sol["meta"]["E"]:
            if sol["se"][e.idx, t] > 0.5:
                closed.add(e.idx)
        for (name, k, tt), v in sol["z"].items():
            if tt == t and v > 0.5:
                refs.add(name)
    def path(e):
        (x1, y1), (x2, y2) = POS[e.i], POS[e.j]
        if e.normally_open and abs(y1 - y2) < 1e-9:          # tie along the same row: draw an arc
            s_ = np.linspace(0, 1, 30)
            return x1 + (x2 - x1) * s_, y1 + 0.75 * np.sin(np.pi * s_)
        return np.array([x1, x2]), np.array([y1, y2])
    for e in NET.edges:
        (x1, y1), (x2, y2) = POS[e.i], POS[e.j]
        xs, ys = path(e)
        if e.idx in C.FAULTED:
            ax.plot([x1, x2], [y1, y2], color="#cc0000", lw=1.0, ls=":")
            ax.plot((x1 + x2) / 2, (y1 + y2) / 2, "x", color="#cc0000", ms=6, mew=1.6)
            continue
        if e.idx in closed:
            col = ISL_COL[bus_isl[e.i] % 4]
            ax.plot(xs, ys, color=col, lw=2.4, solid_capstyle="round", zorder=1)
        else:
            ax.plot(xs, ys, color="#bbbbbb", lw=0.9, ls="--" if e.normally_open else "-", zorder=0)
        if e.switchable:
            ax.plot(xs[len(xs) // 2] if len(xs) > 2 else (x1 + x2) / 2,
                    ys[len(ys) // 2] if len(ys) > 2 else (y1 + y2) / 2, "s", ms=3.2, mfc="white",
                    mec="k" if e.idx in closed else "#999999", mew=0.7, zorder=3)
    for b, (x, y) in POS.items():
        on = b in bus_isl
        tier = NET.loads[b].tier if b in NET.loads else 3
        served = any(r["sol"]["d"].get((b, t), 0) > 1e-6 for r in results)
        fc = ISL_COL[bus_isl[b] % 4] if on else "white"
        ax.scatter(x, y, s=34 if tier == 1 else 22, c=fc, edgecolors="k" if served else "#888888",
                   linewidths=1.4 if tier == 1 else 0.6, zorder=4)
        ax.text(x, y - 0.42, str(b), ha="center", va="top", fontsize=5.2, color="#333333")
    for g in NET.resources:
        if g.cls == "V":
            continue
        x, y = POS[g.bus]
        mk = "*" if g.name in refs else ("^" if g.cls in "DB" else "v")
        ax.scatter(x + 0.28, y + 0.32, marker=mk, s=60 if mk == "*" else 22,
                   c="gold" if mk == "*" else "#444444", edgecolors="k", linewidths=0.4, zorder=5)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.6, 17.8); ax.set_ylim(-3.4, 2.0)
    ax.set_aspect("equal"); ax.axis("off")


def fig_maps():
    cfgs = [("ICA candidates", C.load("ica")), ("ICA + coalition rule (25)", C.load("coalition")["results"]),
            ("Centralized multi-island MILP", C.load("central"))]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.6))
    for ax, (n, r) in zip(axes, cfgs):
        draw_network(ax, r, 1, n + " (interval 1)")
    leg = [Line2D([], [], color="#1b9e77", lw=2.4, label="closed edge (island colour)"),
           Line2D([], [], color="#bbbbbb", lw=0.9, label="open line"),
           Line2D([], [], color="#bbbbbb", lw=0.9, ls="--", label="open tie switch"),
           Line2D([], [], color="#cc0000", ls=":", marker="x", label="faulted / lost supply"),
           Line2D([], [], marker="s", ls="", mfc="white", mec="k", ms=4, label="switch"),
           Line2D([], [], marker="*", ls="", mfc="gold", mec="k", ms=9, label="grid-forming reference"),
           Line2D([], [], marker="^", ls="", color="#444444", ms=5, label="other DG/BESS"),
           Line2D([], [], marker="v", ls="", color="#444444", ms=5, label="renewable"),
           Line2D([], [], marker="o", ls="", mfc="white", mec="k", mew=1.4, ms=6, label="Tier-1 bus (dark edge = served)")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig03_island_maps.png")


# ------------------------------------------------------------------ dispatch and state of charge
def fig_dispatch():
    res = C.load("coalition")["results"]
    sol = res[0]["sol"]; meta = sol["meta"]
    groups = {"Diesel G1": [], "Microturbine G2": [], "PV": [], "Wind": [], "BESS discharge": [], "EV discharge": []}
    charge = np.zeros(H); load = np.zeros(H); loss = np.zeros(H)
    for t in range(1, H + 1):
        groups["Diesel G1"].append(sol["P"].get(("G1-diesel", t), 0))
        groups["Microturbine G2"].append(sol["P"].get(("G2-microturbine", t), 0))
        groups["PV"].append(sum(sol["P"][g.name, t] for g in meta["R"] if g.name.startswith("PV")))
        groups["Wind"].append(sum(sol["P"][g.name, t] for g in meta["R"] if g.name.startswith("WT")))
        groups["BESS discharge"].append(sum(sol["Pdis"][g.name, t] for g in meta["S"] if g.cls == "B"))
        groups["EV discharge"].append(sum(sol["Pdis"][g.name, t] for g in meta["S"] if g.cls == "V"))
        charge[t - 1] = sum(sol["Pch"][g.name, t] for g in meta["S"])
        load[t - 1] = sum(v for (i, tt), v in sol["d"].items() if tt == t)
        loss[t - 1] = sum(sol["ell"][e.idx, t] for e in meta["E"])
    cols = ["#4d4d4d", "#878787", "#fdb863", "#80cdc1", "#5e3c99", "#b2abd2"]
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    bottom = np.zeros(H)
    for (n, v), c in zip(groups.items(), cols):
        v = np.array(v) * Sb
        ax.bar(TMIN + 2.5, v, 4.2, bottom=bottom, color=c, label=n)
        bottom += v
    if charge.max() > 1e-6:
        ax.bar(TMIN + 2.5, -charge * Sb, 4.2, color="#e08214", label="storage charging")
    ax.step(np.r_[TMIN, 60], np.r_[load, load[-1]] * Sb, where="post", color="k", lw=1.4, label="served load")
    ax.step(np.r_[TMIN, 60], np.r_[load + loss + charge, (load + loss + charge)[-1]] * Sb, where="post",
            color="k", lw=0.8, ls="--", label="load + charging + est. loss")
    ax.set_xlabel("time after isolation (min)"); ax.set_ylabel("power (kW)")
    ax.legend(fontsize=6.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    save(fig, "fig04_dispatch.png")


def fig_soc():
    res = C.load("coalition")["results"]
    sol = res[0]["sol"]; meta = sol["meta"]
    tb = np.arange(H + 1) * 5
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for g, c in zip([g for g in meta["S"] if g.cls == "B"], ["#5e3c99", "#1b7837"]):
        axes[0].plot(tb, [sol["SOC"][g.name, t] for t in range(1, H + 2)], "-o", ms=2.5, color=c, label=g.name)
    axes[0].axhline(0.2, color="grey", ls=":", lw=0.8); axes[0].text(1, 0.21, "SoC$^{min}$", fontsize=7)
    axes[0].set_title("Stationary BESS", fontsize=9); axes[0].set_ylabel("state of charge")
    axes[0].set_xlabel("time (min)"); axes[0].legend(fontsize=7)
    for g in [g for g in meta["S"] if g.cls == "V"]:
        info = meta["ev_info"][g.name]
        n_e = min(info["n_e"], H + 1)
        soc = [sol["SOC"][g.name, t] for t in range(1, n_e + 1)]
        axes[1].plot(tb[:n_e], soc, "-", lw=1.0, color="#b2abd2")
        axes[1].plot(tb[n_e - 1], soc[-1], "o", ms=3.5, mfc="#5e3c99" if info["n_e"] <= H + 1 else "white",
                     mec="#5e3c99")
    axes[1].axhline(0.6, color="#b2182b", ls="--", lw=0.9)
    axes[1].text(1, 0.605, "required SoC at departure", fontsize=7, color="#b2182b")
    axes[1].set_title("EVs (filled: departs in horizon; open: end of horizon)", fontsize=8)
    axes[1].set_xlabel("time (min)")
    fig.tight_layout()
    save(fig, "fig05_soc.png")


# ------------------------------------------------------------------ voltages: LinDistFlow vs AC
def ac_voltages(sol, t):
    meta = sol["meta"]
    G = checks.energized_graph(sol, meta, t)
    refs = checks.reference_buses(sol, meta, t)
    V = {}
    for comp in nx.connected_components(G):
        root = next(b for b in comp if b in refs)
        ref = refs[root][0]
        Snet = {}
        for i in comp:
            Pd = sol["d"].get((i, t), 0.0)
            Snet[i] = complex(Pd, Pd * meta["tanphi"].get(i, 0.0))
        for g in meta["res"]:
            if g.bus in comp and g is not ref and sol["u"].get((g.name, t), 0) > 0.5:
                p = sol["P"][g.name, t] if g.cls in "DR" else sol["Pdis"][g.name, t] - sol["Pch"][g.name, t]
                Snet[g.bus] -= complex(p, sol["Q"].get((g.name, t), 0.0))
        Vc, *_ = checks._sweep(G, comp, root, Snet, math.sqrt(sol["v"][root, t]))
        V.update({i: abs(v) for i, v in Vc.items()})
    return V


def fig_voltage():
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    err = []
    for (name, res), c, off in zip([("ICA + coalition", C.load("coalition")["results"]), ("Centralized", C.load("central"))],
                                   ["#1b9e77", "#7570b3"], (-0.15, 0.15)):
        sol = res[0]["sol"]
        for t, mk in ((1, "o"), (12, "^")):
            V = ac_voltages(sol, t)
            buses = sorted(V)
            ax.plot(np.array(buses) + off, [V[b] for b in buses], ls="", marker=mk, ms=3.2, color=c,
                    mfc=c if t == 1 else "white", label=f"{name}, AC, interval {t}")
            err += [abs(V[b] - math.sqrt(sol["v"][b, t])) for b in buses]
        Vl = {b: math.sqrt(sol["v"][b, 1]) for b in sorted(ac_voltages(sol, 1))}
        ax.plot(np.array(list(Vl)) + off, list(Vl.values()), ls="", marker="_", ms=7, mew=1.2, color="k",
                label="LinDistFlow, interval 1" if name.startswith("ICA") else None)
    ax.axhline(0.95, color="#cc0000", lw=0.9, ls=":")
    ax.text(0.6, 0.9503, "$V^{min}$", fontsize=7, color="#cc0000")
    ax.set_ylim(0.948, 0.962); ax.set_xlim(0, 34); ax.set_xlabel("bus"); ax.set_ylabel("voltage (p.u.)")
    ex = int(math.floor(math.log10(max(err))))
    ax.set_title(f"max |AC $-$ LinDistFlow| over both intervals = {max(err) / 10 ** ex:.1f}$\\times10^{{{ex}}}$ p.u.", fontsize=8)
    ax.legend(fontsize=6.3, ncol=3, loc="upper left")
    save(fig, "fig06_voltage.png")


# ------------------------------------------------------------------ Monte Carlo shortfall
def fig_mc():
    det = C.z_samples(C.load("coalition")["results"], "indep", return_samples=True)
    cc = C.z_samples(C.load("cc"), "indep", return_samples=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharex=True)
    t_sel = 6
    for ax, (n, d, c) in zip(axes, [("deterministic", det, "#d95f02"), ("chance-constrained, $\\epsilon$=0.05", cc, "#1b9e77")]):
        Z = next(z for t, z in d["samples"] if t == t_sel)
        ax.hist(Z, bins=80, color=c, alpha=0.8, density=True)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{n}: Pr(Z>0) = {np.mean(Z > 0):.3f}", fontsize=8.5)
        ax.set_xlabel("supply shortfall $Z_{k,t}$ (kW), interval 6")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    save(fig, "fig07_mc_shortfall.png")
    # violation by interval
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    for name, res, c in (("deterministic", C.load("coalition")["results"], "#d95f02"),
                         ("chance-constrained", C.load("cc"), "#1b9e77")):
        for m, ls in (("indep", "-"), ("adverse", "--"), ("student_t", ":")):
            d = C.z_samples(res, m, return_samples=True)
            p = [np.mean(z > 0) for t, z in sorted(d["samples"], key=lambda q: q[0])]
            ax.plot(TMIN + 2.5, np.array(p) * 100, ls, marker="o", ms=2.5, color=c,
                    label=f"{name}, {m.replace('_', '-')}")
    ax.axhline(5, color="k", lw=0.8, ls="-."); ax.text(1, 6.5, "$\\epsilon$ = 5%", fontsize=7)
    ax.set_xlabel("time (min)"); ax.set_ylabel("Pr(Z > 0) (%)")
    ax.legend(fontsize=6, ncol=2, loc="center right")
    save(fig, "fig08_mc_by_interval.png")


# ------------------------------------------------------------------ ablation and coalition sequence
def fig_ablation():
    S = json.load(open(os.path.join(C.OUT, "case33_summary.json")))
    cfgs = [("ICA", "ica"), ("Coalition", "coalition"), ("Central.", "central"),
            ("V6 norm.", "ablation"), ("Chance-c.", "cc")]
    stor = {"cc": 79.2}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    x = np.arange(len(cfgs))
    bottom = np.zeros(len(cfgs))
    for tier in ("1", "2", "3"):
        v = np.array([S[k]["served_kWh_by_tier"][tier] for _, k in cfgs])
        axes[0].bar(x, v, 0.6, bottom=bottom, color=TIER_COL[int(tier)], label=f"Tier {tier}")
        bottom += v
    for xi, (_, k) in zip(x, cfgs):
        axes[0].text(xi, bottom[xi] + 25, f"{S[k]['RL_pct']:.1f}%", ha="center", fontsize=7)
    axes[0].set_xticks(x, [c for c, _ in cfgs], fontsize=7)
    axes[0].set_ylabel("served energy in 1 h (kWh)"); axes[0].legend(fontsize=7)
    sd = [S[k].get("storage_discharge_kWh") or stor.get(k, 0) for _, k in cfgs]
    shed = [S[k]["ac"]["shed_kWh"] for _, k in cfgs]
    axes[1].bar(x - 0.18, sd, 0.36, color="#5e3c99", label="storage discharge")
    axes[1].bar(x + 0.18, shed, 0.36, color="#cc0000", label="AC-validation shedding")
    for xi, v in zip(x, sd):
        if v < 1:
            axes[1].text(xi - 0.18, 8, "0", ha="center", fontsize=7, color="#5e3c99")
    axes[1].set_xticks(x, [c for c, _ in cfgs], fontsize=7, rotation=30)
    axes[1].set_ylabel("energy (kWh)"); axes[1].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig09_configurations.png")


def fig_coalition():
    log = [x for x in C.load("coalition")["log"] if "a" in x]
    labels = []
    for x in log:
        a = "+".join(s.split("-")[0] for s in x["a"]); b = "+".join(s.split("-")[0] for s in x["b"])
        labels.append(f"{a} | {b}")
    rounds = [1] * 5 + [2] * 2 + [3] * 1
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    cols = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3"}
    y = np.arange(len(log))[::-1]
    acc = {0, 5, 7}       # largest dJ >= theta_J in each round (merged)
    ax.barh(y, [x["dJ"] for x in log], color=[cols[r] for r in rounds],
            edgecolor=["k" if q in acc else "none" for q in range(len(log))], linewidth=1.2)
    ax.axvline(C.THETA_J, color="k", ls="--", lw=0.8); ax.text(C.THETA_J + 0.003, len(log) - 0.6, "$\\theta_J$", fontsize=8)
    for yi, x in zip(y, log):
        q = len(log) - 1 - yi
        lim = " time limit" if x["solve_s"] > 119 else ""
        ax.text(x["dJ"] + 0.003, yi, f"{x['dJ']:.3f}  ({x['solve_s']:.0f} s{lim})" + ("  merged" if q in acc else ""),
                va="center", fontsize=6.5, fontweight="bold" if q in acc else "normal")
    ax.set_yticks(y, labels, fontsize=7); ax.set_xlabel("$\\Delta J$ of rule (25)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cols[r]) for r in (1, 2, 3)],
              labels=["round 1", "round 2", "round 3"], fontsize=7, loc="lower right")
    ax.set_xlim(0, 0.32)
    save(fig, "fig10_coalition_rounds.png")


# ------------------------------------------------------------------ verification panel
def fig_verification():
    V = json.load(open(os.path.join(C.OUT, "verification.json")))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    # V3 polygon
    ax = axes[0]; Mp = 12
    ax.add_patch(Circle((0, 0), 1, fill=False, color="k", lw=1.0))
    for c, col, lab in ((math.cos(math.pi / Mp), "#1b9e77", "with cos($\\pi/M_p$)"), (1.0, "#d95f02", "tangent")):
        r = c / math.cos(math.pi / Mp)
        ang = 2 * np.pi * np.arange(1, Mp + 1) / Mp + math.pi / Mp
        ax.add_patch(MplPolygon(np.c_[r * np.cos(ang), r * np.sin(ang)], fill=False, color=col, lw=1.0, label=lab))
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
    ax.set_title("V3: rating polygon (7)", fontsize=8.5); ax.legend(fontsize=6, loc="lower left")
    ax.text(0, 0.1, "max |S|/S$^{max}$\n1.000 vs 1.035", ha="center", fontsize=6.5)
    ax.set_xlabel("$P/S^{max}$"); ax.set_ylabel("$Q/S^{max}$")
    # V4
    ax = axes[1]; d = V["V4"]
    labs = ["single\nisland", "multi-\nisland", "without\n(14)"]
    keys = ["single_island", "multi_island", "without_flow_14"]
    ok = [d[k]["all_components_valid"] for k in keys]; bad = [d[k]["instances"] - d[k]["all_components_valid"] for k in keys]
    ax.bar(labs, ok, color="#1b9e77", label="valid forest, one root/comp.")
    ax.bar(labs, bad, bottom=ok, color="#cc0000", label="cycle / wrong root count")
    ax.set_title("V4: Proposition 1", fontsize=8.5); ax.set_ylabel("instances"); ax.legend(fontsize=6)
    ax.set_ylim(0, 190)
    # V5
    ax = axes[2]; s = V["V5"]["solved"]
    t = np.arange(1, 7)
    ax.bar(t - 0.2, s["ev_discharge_kW"], 0.4, color="#5e3c99", label="EV discharge (kW)")
    ax.bar(t + 0.2, s["served_kW"], 0.4, color="#67a9cf", label="served load (kW)")
    ax.axvline(2.5, color="k", ls="--", lw=0.8); ax.text(2.6, 17, "departure\n(boundary 3)", fontsize=6.5)
    ax.set_xticks(t); ax.set_xlabel("interval"); ax.set_title("V5: EV departure (10)", fontsize=8.5)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    save(fig, "fig11_verification.png")


# ------------------------------------------------------------------ sensitivity figures
def fig_sensitivity():
    path = os.path.join(C.OUT, "sensitivity.json")
    if not os.path.exists(path):
        return
    S = json.load(open(path))
    if S.get("df"):
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
        for cfg, c in (("ica", "#d95f02"), ("coalition", "#1b9e77")):
            r = sorted([x for x in S["df"] if x["config"] == cfg], key=lambda q: q["df"])
            df = [100 * x["df"] for x in r]
            lab = "ICA" if cfg == "ica" else "ICA + coalition"
            axes[0].plot(df, [x["RL"] for x in r], "-o", ms=3, color=c, label=lab)
            axes[1].plot(df, [x["RC"] for x in r], "-o", ms=3, color=c, label=lab)
        r = sorted([x for x in S["df"] if x["config"] == "ica"], key=lambda q: q["df"])
        df = np.array([100 * x["df"] for x in r]); bottom = np.zeros(len(r))
        for tier in ("1", "2", "3"):
            v = np.array([x["tier"][tier] for x in r])
            axes[2].bar(np.arange(len(r)), v, 0.6, bottom=bottom, color=TIER_COL[int(tier)], label=f"Tier {tier}")
            bottom += v
        axes[2].set_xticks(np.arange(len(r)), [f"{d:g}" for d in df]); axes[2].set_title("ICA: served energy by tier", fontsize=8)
        axes[2].set_ylabel("kWh in 1 h"); axes[2].legend(fontsize=6)
        axes[0].set_ylabel("restored energy RL (%)"); axes[1].set_ylabel("critical-load restoration RC (%)")
        for ax in axes:
            ax.set_xlabel("$\\Delta f^{max}$ (% of $f_0$)")
        for ax in axes[:2]:
            ax.axvline(1, color="grey", ls=":", lw=0.8)
        axes[0].legend(fontsize=6.5)
        fig.tight_layout()
        save(fig, "fig12_df_sensitivity.png")
    if S.get("eps"):
        r = sorted([x for x in S["eps"] if x["eps"] is not None], key=lambda q: q["eps"])
        det = next((x for x in S["eps"] if x["eps"] is None), None)
        e = np.array([100 * x["eps"] for x in r])
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
        ax = axes[0]
        ax.plot(e, [x["RL"] for x in r], "-o", ms=3.5, color="#1b9e77", label="chance-constrained (27)")
        for xi, x in zip(e, r):
            if x["gap"] > 0.02:
                ax.annotate(f"gap {100 * x['gap']:.1f}%", (xi, x["RL"]), fontsize=6, xytext=(-14, 6), textcoords="offset points")
        if det:
            ax.axhline(det["RL"], color="#d95f02", ls="--", lw=1.0, label="deterministic")
        ax.set_xscale("log"); ax.set_xticks(e, [f"{v:g}" for v in e])
        ax.set_xlabel("risk level $\\epsilon$ (%)"); ax.set_ylabel("restored energy RL (%)")
        ax.set_ylim(0, 52); ax.legend(fontsize=6.5, loc="lower right")
        ax = axes[1]
        floor = 1e-3
        for m, mk, lab in (("indep", "o", "independent"), ("adverse", "s", "adverse"), ("student_t", "^", "Student-t")):
            v = np.array([100 * x["mc"][m]["max"] for x in r])
            ax.plot(e, np.maximum(v, floor), marker=mk, ms=3.5, lw=0.9, label=lab)
        ax.plot([0.8, 25], [0.8, 25], color="k", lw=0.9, label="Pr(Z>0) = $\\epsilon$")
        if det:
            ax.axhline(100 * max(det["mc"][m]["max"] for m in det["mc"]), color="#d95f02", ls="--", lw=1.0,
                       label="deterministic (max)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(e, [f"{v:g}" for v in e])
        ax.set_ylim(5e-4, 150); ax.axhspan(5e-4, 1.4e-3, color="grey", alpha=0.12)
        ax.text(1.05, 6.5e-4, "no violation in $2\\times10^4$ samples", fontsize=6)
        ax.set_xlabel("risk level $\\epsilon$ (%)"); ax.set_ylabel("max empirical Pr(Z>0) (%)")
        ax.legend(fontsize=6, loc="center right")
        fig.tight_layout()
        save(fig, "fig13_eps_tradeoff.png")
    if S.get("scale"):
        r = sorted(S["scale"], key=lambda q: q["buses"])
        fig, ax = plt.subplots(figsize=(4.6, 2.6))
        rc = {1: "#1b9e77", 2: "#d95f02", 3: "#7570b3", 4: "#e7298a"}
        for nref in sorted({x["refs"] for x in r}):
            q = [x for x in r if x["refs"] == nref]
            ax.scatter([x["buses"] for x in q], [max(x["solve_s"], 0.01) for x in q], s=22, color=rc[nref],
                       label=f"single island, {nref} eligible reference{'s' if nref > 1 else ''}", zorder=3)
        for x in r:
            if x["status"] != "Optimal":
                ax.annotate(f"time limit, gap {100 * x['gap']:.1f}%", (x["buses"], x["solve_s"]), fontsize=6,
                            xytext=(-62, 6), textcoords="offset points")
        cen = C.load("central")[0]["info"]
        ax.scatter([32], [cen["solve_s"]], marker="*", s=90, color="#cc0000",
                   label="centralized, 4 potential islands", zorder=4)
        ax.annotate(f"time limit, gap {100 * cen['mip_gap']:.1f}%", (32, cen["solve_s"]), fontsize=6,
                    xytext=(-80, -3), textcoords="offset points")
        ax.set_yscale("log"); ax.set_xlabel("candidate island size (buses)"); ax.set_ylabel("solve time (s)")
        ax.legend(fontsize=6, loc="lower right")
        save(fig, "fig14_scalability.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "base"):
        fig_maps(); fig_dispatch(); fig_soc(); fig_voltage(); fig_mc(); fig_ablation(); fig_coalition(); fig_verification()
    if which in ("all", "sens"):
        fig_sensitivity()
