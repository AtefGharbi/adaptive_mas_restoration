"""Sensitivity runs for Section X-J (Figs. 6-8).  Each point is saved as soon as it is solved.

  df      frequency-deviation limit Delta f^max in {0.5,1,1.5,2,3}% for the ICA partition and
          for the merged coalition island
  eps     chance-constraint risk level eps in {0.01,0.02,0.05,0.1,0.2} on the coalition island,
          with Monte Carlo violation frequency
  scale   solve time and model size versus candidate size (BFS shells seeded at bus 7)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import case33 as C  # noqa: E402
from restoration.candidates import bfs_shells  # noqa: E402

OUT = os.path.join(C.OUT, "sensitivity.json")


def store(key, val):
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d.setdefault(key, []).append(val)
    json.dump(d, open(OUT, "w"), indent=1)


def done(key, tag):
    if not os.path.exists(OUT):
        return False
    return any(r.get("tag") == tag for r in json.load(open(OUT)).get(key, []))


def summarize(res, tag, extra=None):
    ev = C.evaluate(res)
    out = dict(tag=tag, RL=ev["RL_pct"], RC=ev["RC_pct"], ENS=ev["ENS_kWh"], N_sw=ev["N_sw"],
               solve_max=ev["solve_s_max"], solve_sum=ev["solve_s_sum"], gap=ev["max_gap"],
               ac_ok=ev["ac"]["feasible"], tier=ev["served_kWh_by_tier"])
    out.update(extra or {})
    return out


def run_df():
    coal_isl = [r["isl"] for r in C.load("coalition")["results"]]
    part = [r["isl"] for r in C.load("ica")]
    for df in (0.005, 0.01, 0.015, 0.02, 0.03):
        for name, isl in (("ica", part), ("coalition", coal_isl)):
            tag = f"{name}|{df}"
            if done("df", tag):
                continue
            res = C.stage_ica(params=dict(dfmax_frac=df, time_limit=60), tag=tag, partition=isl)
            store("df", summarize(res, tag, dict(config=name, df=df)))


def run_eps():
    coal_isl = [r["isl"] for r in C.load("coalition")["results"]]
    for eps in (0.01, 0.02, 0.05, 0.1, 0.2):
        tag = f"eps|{eps}"
        if done("eps", tag):
            continue
        res = C.stage_ica(params=dict(eps=eps, time_limit=60), tag=tag, partition=coal_isl)
        mc = {m: C.z_samples(res, m) for m in ("indep", "adverse", "student_t")}
        store("eps", summarize(res, tag, dict(eps=eps, mc=mc)))
    tag = "eps|det"
    if not done("eps", tag):
        res = C.load("coalition")["results"]
        mc = {m: C.z_samples(res, m) for m in ("indep", "adverse", "student_t")}
        store("eps", summarize(res, tag, dict(eps=None, mc=mc)))


def run_scale():
    shells = []
    for seed in (7, 30, 18, 24):            # shells seeded at each eligible reference
        shells += bfs_shells(C.NET, C.OOS, C.FAULTED, seed)
    seen = set()
    for C_k in sorted(shells, key=len):
        key = len(C_k)
        if key in seen:
            continue
        seen.add(key)
        tag = f"scale|{key}"
        if done("scale", tag):
            continue
        refs = {g.name for g in C.NET.resources if g.bus in C_k and g.GFM and g.BS}
        _, sol, info = C.run_island([{"C": set(C_k), "refs": refs}], params=dict(time_limit=60))
        store("scale", dict(tag=tag, buses=key, refs=len(refs), nvars=info["nvars"], ncons=info["ncons"],
                            solve_s=info["solve_s"], gap=info["mip_gap"], status=info["highs_status"]))
        print(tag, info["highs_status"], round(info["solve_s"], 1))


if __name__ == "__main__":
    {"df": run_df, "eps": run_eps, "scale": run_scale}[sys.argv[1]]()
