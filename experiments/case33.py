"""Illustrative IEEE 33-bus case study (paper Section X-J, Tables IX-XI).

Stages (run in order; each caches its output in results/):
  ica        Stage 3/5: nearest-reference candidates, one single-island problem (23) per ICA
  coalition  Stage 6: feasibility-first coalition rule (25), greedy pairwise merging
  central    B1-type multi-island MILP over the whole area (island-indexed formulation)
  ablation   coalition configuration with the V6 capacity-based normalization of (21)-(22)
  cc         chance-constrained (27) vs deterministic, Monte Carlo under three error models
  trip       Scenario 7: G1 trips at minute 30; Stage 8 re-solve with switching history
  report     collect everything into results/case33_summary.json and figures

Event: loss of upstream supply (edge 1-2 open) plus a permanent fault on line 12-13.
All 32 load buses are out of service.  Horizon: H = 12 intervals of 5 min.
"""
import json
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from restoration.data import ieee33  # noqa: E402
from restoration.model import Case, solve, compute_norm, load_forecast, renewable_forecast, clipped_moments  # noqa: E402
from restoration.candidates import nearest_reference_partition, eligible_refs, sections, adequacy_screen  # noqa: E402
from restoration import checks  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
H, DT = 12, 5 / 60
OOS = set(range(2, 34))
FAULTED = {0, 11}            # edge indices of lines 1-2 and 12-13
THETA_J = 0.01
NET = ieee33(H=24)


def base_case(**kw):
    return Case(NET, H=kw.pop("H", H), dt=DT, islands=[{"C": set(OOS), "refs": set()}], oos=OOS,
                faulted=FAULTED, **kw)


NORM = compute_norm(base_case())


def norm_for(params):
    if params.get("norm_mode") == "capacity":
        return compute_norm(base_case(params=dict(norm_mode="capacity")))
    return NORM


def run_island(isl, params=None, **kw):
    params = {"time_limit": 120, **(params or {})}
    c = Case(NET, H=kw.pop("H", H), dt=DT, islands=isl, oos=OOS, faulted=FAULTED,
             norm=kw.pop("norm", norm_for(params)), params=params, **kw)
    sol, info, _ = solve(c)
    return c, sol, info


def save(name, obj):
    with open(os.path.join(OUT, f"case33_{name}.pkl"), "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(os.path.join(OUT, f"case33_{name}.pkl"), "rb") as f:
        return pickle.load(f)


def strip(sol):
    """Remove nothing heavy; Resource objects in meta are picklable dataclasses."""
    return sol


# ------------------------------------------------------------------------------------------
def adjacent(Ca, Cb):
    for e in NET.edges:
        if e.idx in FAULTED or not e.switchable:
            continue
        if (e.i in Ca and e.j in Cb) or (e.i in Cb and e.j in Ca):
            return True
    return False


def stage_ica(params=None, tag="ica", partition=None, **kw):
    part = partition or nearest_reference_partition(NET, OOS, FAULTED, kw.get("unavailable", ()))
    res = []
    for isl in part:
        c, sol, info = run_island([isl], params=params, **kw)
        res.append(dict(isl=isl, sol=sol, info=info,
                        screen_full_pickup=adequacy_screen(c, isl["C"], kw.get("unavailable", ()))))
        print(tag, sorted(isl["refs"]), len(isl["C"]), info["highs_status"], round(info["solve_s"], 2),
              round(info["obj"], 4))
    return res


def stage_coalition(start, params=None, tag="coal", **kw):
    """Greedy feasibility-first merging with rule (25)."""
    cur = list(start)
    log = []
    while True:
        best = None
        for a in range(len(cur)):
            for b in range(a + 1, len(cur)):
                A, B = cur[a], cur[b]
                if not adjacent(A["isl"]["C"], B["isl"]["C"]):
                    continue
                merged = {"C": A["isl"]["C"] | B["isl"]["C"], "refs": A["isl"]["refs"] | B["isl"]["refs"]}
                c, sol, info = run_island([merged], params=params, **kw)
                feas = sol is not None and info["highs_status"] in ("Optimal", "Time limit reached")
                dJ = (info["obj"] - A["info"]["obj"] - B["info"]["obj"]) if feas else None
                log.append(dict(a=sorted(A["isl"]["refs"]), b=sorted(B["isl"]["refs"]), feasible=feas,
                                dJ=dJ, solve_s=info["solve_s"], gap=info["mip_gap"]))
                print(tag, "pair", sorted(A["isl"]["refs"]), sorted(B["isl"]["refs"]),
                      None if dJ is None else round(dJ, 4), round(info["solve_s"], 1))
                if feas and dJ >= THETA_J and (best is None or dJ > best[0]):
                    best = (dJ, a, b, dict(isl=merged, sol=sol, info=info))
        if best is None:
            break
        _, a, b, new = best
        cur = [x for q, x in enumerate(cur) if q not in (a, b)] + [new]
        log.append(dict(merged=sorted(new["isl"]["refs"]), dJ=best[0]))
    return cur, log


# ------------------------------------------------------------------------------------------
def evaluate(results, net=NET, label=""):
    """Metrics of Section X-F over the 1-h horizon, from AC-validated served load."""
    Sb = net.S_base
    aff = NORM["aff"]
    t1 = [i for i in aff if net.loads[i].tier == 1]
    served = {}                 # (i, t) -> p.u. after AC validation
    ac = dict(feasible=True, n_redispatch=0, n_shed=0, shed_kWh=0.0, ac_loss=0.0, model_loss=0.0,
              vmin=9.0, vmax=0.0, max_loading=0.0)
    graph_ok, nsw, solve_s, gaps, dwell_bad, ev_ok = True, 0, [], [], 0, True
    for r in results:
        sol, info = r["sol"], r["info"]
        solve_s.append(info["solve_s"])
        gaps.append(info["mip_gap"] or 0.0)
        meta = sol["meta"]
        graph_ok &= all(x["ok"] for x in checks.graph_check(sol, meta))
        rep = checks.ac_validate(sol, meta, net)
        for k in ("n_redispatch", "n_shed", "shed_kWh", "ac_loss", "model_loss"):
            ac[k] += rep[k]
        ac["feasible"] &= rep["feasible"]
        if rep["n_islands"]:
            ac["vmin"] = min(ac["vmin"], rep["vmin"]); ac["vmax"] = max(ac["vmax"], rep["vmax"])
            ac["max_loading"] = max(ac["max_loading"], rep["max_loading"])
        served.update(rep["d_eval"])
        nsw += int(round(sum(sol["om"].values())))
        dwell_bad += checks.dwell_violations(sol, meta, 3, r.get("hist_s", {}))
        for g in meta["V"]:
            info_e = meta["ev_info"].get(g.name)
            if info_e and info_e["n_e"] <= H + 1:
                ev_ok &= sol["SOC"][g.name, info_e["n_e"]] >= info_e["req_eff"] - 1e-6
    T = range(1, H + 1)
    Eall = sum(load_forecast(base_case(), i, t) * DT for i in aff for t in T)
    ET1 = sum(load_forecast(base_case(), i, t) * DT for i in t1 for t in T)
    Es = sum(served.get((i, t), 0) * DT for i in aff for t in T)
    Es1 = sum(served.get((i, t), 0) * DT for i in t1 for t in T)
    by_tier = {tier: round(sum(served.get((i, t), 0) * DT for i in aff if net.loads[i].tier == tier
                               for t in T) * Sb, 1) for tier in (1, 2, 3)}
    first = [t for t in T if any(served.get((i, t), 0) > 1e-6 for i in aff)]
    allt1 = [t for t in T if all(served.get((i, t), 0) > 1e-6 for i in t1)]
    out = dict(label=label, RL_pct=100 * Es / Eall, RC_pct=100 * Es1 / ET1, ENS_kWh=(Eall - Es) * Sb,
               served_kWh_by_tier=by_tier, served_kW_at_end=round(sum(served.get((i, H), 0) for i in aff) * Sb, 1),
               T1_buses_restored=sorted(i for i in t1 if served.get((i, H), 0) > 1e-6),
               N_sw=nsw, T_init_min=(first[0] - 1) * 5 if first else None,
               T_crit_min=(allt1[0] - 1) * 5 if allt1 else None,
               n_islands=len(results), solve_s_sum=sum(solve_s), solve_s_max=max(solve_s),
               max_gap=max(gaps), graph_ok=graph_ok, dwell_violations=dwell_bad, ev_departure_ok=ev_ok,
               ac=ac, served=served)
    return out


# ------------------------------------------------------------------------------------------
def z_samples(results, model="indep", N=20000, seed=11, sigma_load=0.10, return_samples=False):
    """Monte Carlo of Z_{k,t} for every energized island/interval of a configuration."""
    rng = np.random.default_rng(seed)
    Sb = NET.S_base
    viol, samples = [], []
    for r in results:
        sol = r["sol"]; meta = sol["meta"]
        for t in meta["T"]:
            G = checks.energized_graph(sol, meta, t)
            import networkx as nx
            for comp in nx.connected_components(G):
                dem = sum(sol["d"].get((i, t), 0) for i in comp)
                ch = sum(sol["Pch"][g.name, t] for g in meta["S"] if g.bus in comp)
                loss = sum(sol["ell"][e.idx, t] for e in meta["E"] if e.i in comp and e.j in comp)
                kk = next(k for (b, k, tt), v in sol["y"].items() if tt == t and b in comp and v > 0.5)
                step = sol["dP"][kk, t]
                capD = sum(g.Pmax / Sb * sol["u"][g.name, t] for g in meta["D"] if g.bus in comp)
                loads = [(load_forecast(base_case(), i, t) * sol["x"][i, t]) for i in comp if (i, t) in sol["x"]]
                ren = [(renewable_forecast(base_case(), g, t), g.sigma * g.rated / Sb, g.rated / Sb)
                       for g in meta["R"] if g.bus in comp and sol["u"][g.name, t] > 0.5]
                stor, evs = 0.0, []
                for g in meta["S"]:
                    if g.bus not in comp or sol["u"][g.name, t] < 0.5:
                        continue
                    socres = max(g.socmin, g.socreq) if g.cls == 'V' else g.socmin
                    cap = min(g.Pmax / Sb, g.eta_d * (g.E / Sb) * (sol["SOC"][g.name, t] - socres) / 0.25)
                    cap = max(cap, 0.0)
                    if g.cls == 'V':
                        evs.append((cap, g.pi))
                    else:
                        stor += cap
                if model == "indep":
                    eL = rng.standard_normal((N, len(loads)))
                    eR = rng.standard_normal((N, len(ren)))
                elif model == "adverse":      # common factor: loads up, renewables down
                    w = rng.standard_normal((N, 1))
                    eL = np.repeat(w, len(loads), 1)
                    eR = -np.repeat(w, len(ren), 1)
                else:                          # Student-t, 3 dof, unit variance
                    eL = rng.standard_t(3, (N, len(loads))) / math.sqrt(3)
                    eR = rng.standard_t(3, (N, len(ren))) / math.sqrt(3)
                Z = np.full(N, dem + ch + loss + step - capD - stor)
                for q, Ph in enumerate(loads):
                    Z += sigma_load * Ph * eL[:, q]
                for q, (f, sd, hi) in enumerate(ren):
                    Z -= np.clip(f + sd * eR[:, q], 0, hi)
                for cap, pi in evs:
                    Z -= cap * (rng.random(N) < pi)
                viol.append(float(np.mean(Z > 1e-9)))
                if return_samples:
                    samples.append((t, Z * Sb))
    out = dict(max=max(viol) if viol else 0.0, mean=float(np.mean(viol)) if viol else 0.0, n=len(viol))
    if return_samples:
        out["samples"] = samples
    return out


# ------------------------------------------------------------------------------------------
def history_from(results, t_end, n=4):
    """Switch/pick-up states at intervals t_end-n+1..t_end of a previous solve (tau<=0)."""
    hs, hx, soc = {}, {}, {}
    for r in results:
        sol = r["sol"]; meta = sol["meta"]
        for e in meta["E"]:
            if e.switchable:
                hs[e.idx] = [int(round(sol["se"][e.idx, t])) for t in range(t_end - n + 1, t_end + 1)]
        for i in meta["aff"]:
            hx[i] = [int(round(sol["x"][i, t])) for t in range(t_end - n + 1, t_end + 1)]
        for g in meta["S"]:
            soc[g.name] = sol["SOC"][g.name, t_end + 1]
    return hs, hx, soc


def stage_trip(coal):
    """G1 trips at boundary 7 (minute 30).  Its island collapses (safety de-energization, which
    bypasses dwell); the other islands keep their switching history.  Stage 3-6 are re-run
    for the whole area without G1, warm-started from the recorded state."""
    hs, hx, soc = history_from(coal, 6)
    g1_isl = next(r for r in coal if "G1-diesel" in r["isl"]["refs"])
    collapsed = g1_isl["isl"]["C"]
    for e in NET.edges:
        if e.idx in hs and (e.i in collapsed or e.j in collapsed):
            hs[e.idx] = [0, 0, 0, 0]
    for i in list(hx):
        if i in collapsed:
            hx[i] = [0, 0, 0, 0]
    kw = dict(t0=6, hist_s=hs, hist_x=hx, soc_init=soc, unavailable={"G1-diesel"})
    part = nearest_reference_partition(NET, OOS, FAULTED, {"G1-diesel"})
    ica = stage_ica(tag="trip-ica", partition=part, **kw)
    for r in ica:
        r["hist_s"] = hs
    cur, log = stage_coalition(ica, tag="trip-coal", **kw)
    for r in cur:
        r["hist_s"] = hs
    return dict(before=coal, ica=ica, after=cur, log=log, collapsed=sorted(collapsed))


# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()
    if stage == "ica":
        save("ica", stage_ica())
    elif stage == "coalition":
        cur, log = stage_coalition(load("ica"))
        save("coalition", dict(results=cur, log=log))
    elif stage == "central":
        tl = int(sys.argv[2]) if len(sys.argv) > 2 else 900
        isl = [{"C": set(OOS), "refs": {g.name}} for g in eligible_refs(NET, OOS)]
        c, sol, info = run_island(isl, params=dict(time_limit=tl))
        print("central", info)
        save("central", [dict(isl=isl, sol=sol, info=info)])
    elif stage == "ablation":
        ica = stage_ica(params=dict(norm_mode="capacity"), tag="abl-ica")
        cur, log = stage_coalition(ica, params=dict(norm_mode="capacity"), tag="abl-coal")
        save("ablation", dict(results=cur, log=log))
    elif stage == "cc":
        coal = load("coalition")["results"]
        part = [r["isl"] for r in coal]
        cc = stage_ica(params=dict(eps=0.05), tag="cc", partition=part)
        save("cc", cc)
    elif stage == "trip":
        save("trip", stage_trip(load("coalition")["results"]))
    elif stage == "report":
        S = {}
        ica = load("ica"); coal = load("coalition"); S["coalition_log"] = coal["log"]
        S["ica"] = evaluate(ica, label="ICA, nearest-reference candidates")
        S["coalition"] = evaluate(coal["results"], label="ICA + coalition rule (25)")
        if os.path.exists(os.path.join(OUT, "case33_central.pkl")):
            cen = load("central")
            S["central"] = evaluate(cen, label="Centralized multi-island MILP")
            S["central"]["info"] = {k: v for k, v in cen[0]["info"].items()}
        if os.path.exists(os.path.join(OUT, "case33_ablation.pkl")):
            ab = load("ablation")
            S["ablation"] = evaluate(ab["results"], label="ICA + coalition, V6 normalization")
            S["ablation_log"] = ab["log"]
        for key in ("ica", "coalition", "central", "ablation"):
            if key in S:
                res = {"ica": ica, "coalition": coal["results"]}.get(key) or \
                      (load("central") if key == "central" else load("ablation")["results"])
                S[key]["storage_discharge_kWh"] = round(sum(
                    r["sol"]["Pdis"][g.name, t] * DT * NET.S_base for r in res
                    for g in r["sol"]["meta"]["S"] for t in range(1, H + 1)), 1)
        if os.path.exists(os.path.join(OUT, "case33_cc.pkl")):
            cc = load("cc")
            S["cc"] = evaluate(cc, label="ICA + coalition, chance-constrained eps=0.05")
            S["mc"] = {}
            for name, res in (("deterministic", coal["results"]), ("chance_constrained", cc)):
                S["mc"][name] = {m: z_samples(res, m) for m in ("indep", "adverse", "student_t")}
        if os.path.exists(os.path.join(OUT, "case33_trip.pkl")):
            tr = load("trip")
            S["trip_log"] = tr["log"]
            S["trip_collapsed"] = tr["collapsed"]
            after = evaluate(tr["after"], label="after G1 trip (t0 = 30 min)")
            S["trip_after"] = after
            bef = evaluate(tr["before"], label="before")
            S["trip_before_at_t6_by_tier_kW"] = {tier: round(sum(bef["served"].get((i, 6), 0)
                                                 for i in NORM["aff"] if NET.loads[i].tier == tier) * 1000, 1)
                                                 for tier in (1, 2, 3)}
            S["trip_after_by_interval_by_tier_kW"] = {tier: [round(sum(after["served"].get((i, t), 0)
                                                      for i in NORM["aff"] if NET.loads[i].tier == tier) * 1000, 1)
                                                      for t in range(1, H + 1)] for tier in (1, 2, 3)}
        for k in list(S):
            if isinstance(S[k], dict) and "served" in S[k]:
                S[k].pop("served")
        with open(os.path.join(OUT, "case33_summary.json"), "w") as f:
            json.dump(S, f, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o))
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                                 if kk not in ("ac",)})
                          for k, v in S.items()}, indent=1, default=str)[:6000])
    print(f"[{stage}] done in {time.time() - t0:.1f} s")
