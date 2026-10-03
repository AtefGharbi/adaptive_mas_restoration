"""Verification suite for the island model (paper Section X-J, Table VIII).

V1  loss allocation and island balance identity (17)
V2  secant loss estimate (7a): never under-estimates; error for uniform vs quadratic spacing
V3  inscribed branch-rating polygon (7)
V4  Proposition 1 on random meshed networks (single- and multi-island), and a negative test
    with the connectivity flow (14) removed
V5  EV departure indexing (10)
V6  dwell constraint (16) with switching history from a previous solve
"""
import json
import math
import os
import sys

import numpy as np
import pulp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from restoration.data import Edge, Load, Network, Resource, small_test_net  # noqa: E402
from restoration.model import Case, build, solve, secant_breakpoints, extract  # noqa: E402
from restoration import checks  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
R = {}


def v1_identity():
    net = small_test_net()
    c = Case(net, H=3, dt=5 / 60, islands=[{"C": {1, 2}, "refs": {"G"}}], oos={1, 2},
             params=dict(Nsw=1))
    sol, info, _ = solve(c)
    t = 1
    Pg, d, ell, Pf = sol["P"]["G", t], sol["d"][2, t], sol["ell"][0, t], sol["Pf"][0, t]
    R["V1"] = dict(
        two_bus=dict(P_gen_kW=Pg * 1000, load_kW=d * 1000, loss_est_kW=ell * 1000, flow_kW=Pf * 1000,
                     gen_minus_load_minus_loss=Pg - d - ell, flow_minus_load_minus_half_loss=Pf - d - ell / 2),
        residual_17_max=max(abs(r) for t in sol["meta"]["T"]
                            for r in checks.island_balance_residual(sol, sol["meta"], t)))
    print("V1", R["V1"])


def v2_secant():
    out = {}
    for spacing in ("uniform", "quadratic"):
        bp = secant_breakpoints(1.0, 8, spacing)
        xs = np.linspace(-1, 1, 20001)
        chord = np.max(np.stack([(a + b) * xs - a * b for a, b in zip(bp[:-1], bp[1:])]), axis=0)
        under = float(np.max(xs ** 2 - chord))
        rel = {f"{f:.2f}": float((chord[np.argmin(abs(xs - f))] - f ** 2) / f ** 2)
               for f in (0.05, 0.1, 0.25, 0.5, 1.0)}
        out[spacing] = dict(max_underestimate=under, rel_overestimate_at_flow_fraction=rel,
                            max_abs_overestimate=float(np.max(chord - xs ** 2)))
    R["V2"] = out
    print("V2", out)


def v3_polygon(Mp=12):
    """Intersect adjacent facets of (7) numerically and report the largest vertex radius."""
    th = 2 * np.pi * np.arange(1, Mp + 1) / Mp
    out = {}
    for label, c in (("inscribed_cos_factor", math.cos(math.pi / Mp)), ("tangent_no_factor", 1.0)):
        rmax = 0.0
        for m in range(Mp):
            A = np.array([[math.cos(th[m]), math.sin(th[m])],
                          [math.cos(th[(m + 1) % Mp]), math.sin(th[(m + 1) % Mp])]])
            v = np.linalg.solve(A, np.array([c, c]))
            rmax = max(rmax, float(np.hypot(*v)))
        out[label] = dict(max_vertex_S_over_Smax=rmax)
    out["Mp"] = Mp
    out["unused_rating_in_facet_direction"] = 1 - math.cos(math.pi / Mp)
    R["V3"] = out
    print("V3", out)


def random_net(rng, n=8, extra=3, nref=2):
    edges, k = [], 0
    for j in range(2, n + 1):
        i = int(rng.integers(1, j))
        edges.append(Edge(k, i, j, 0.01, 0.01, 3000.0, switchable=True)); k += 1
    pairs = {(e.i, e.j) for e in edges}
    while extra > 0:
        i, j = sorted(rng.choice(np.arange(1, n + 1), 2, replace=False).tolist())
        if (i, j) not in pairs:
            edges.append(Edge(k, i, j, 0.01, 0.01, 3000.0, switchable=True)); k += 1
            pairs.add((i, j)); extra -= 1
    loads = {b: Load(20, 5, tier=3) for b in range(1, n + 1)}
    rb = rng.choice(np.arange(1, n + 1), nref, replace=False)
    res = [Resource(f"G{q}", int(b), 'D', Pmax=300, Pmin=0, Qmax=200, rated=300, droop=0.04,
                    BS=1, GFM=1, FR=1, VR=1, PROT=1, If=2.0) for q, b in enumerate(rb)]
    return Network("rand", list(range(1, n + 1)), edges, loads, res)


def prop1_trial(rng, multi, no_flow=False):
    net = random_net(rng, nref=int(rng.integers(1, 4)))
    oos = set(net.buses)
    if multi:
        isl = [{"C": set(oos), "refs": {g.name}} for g in net.resources]
    else:
        isl = [{"C": set(oos), "refs": {g.name for g in net.resources}}]
    c = Case(net, H=1, dt=5 / 60, islands=isl, oos=oos,
             params=dict(Nsw=1, x_dwell=False, no_conn_flow=no_flow, time_limit=20))
    m, V, meta = build(c)
    # adversarial objective: reward closing edges and energizing buses (pushes toward loops)
    m.setObjective(m.objective + pulp.lpSum(float(rng.uniform(0.5, 2)) * v for v in V["se"].values())
                   + pulp.lpSum(float(rng.uniform(0.1, 1)) * v for v in V["yi"].values()))
    m.solve(pulp.HiGHS(msg=False, timeLimit=20))
    if pulp.LpStatus[m.status] != "Optimal":
        return None
    sol = extract(c, m, V, meta)
    rows = checks.graph_check(sol, meta)
    return all(r["ok"] for r in rows), any(not r["tree"] for r in rows), any(r["nref"] != 1 for r in rows)


def v4_prop1(n_single=150, n_multi=100, n_neg=60):
    rng = np.random.default_rng(2026)
    out = {}
    for label, n, multi, nf in (("single_island", n_single, False, False),
                                ("multi_island", n_multi, True, False),
                                ("without_flow_14", n_neg, False, True)):
        ok = cyc = badref = solved = 0
        for _ in range(n):
            r = prop1_trial(rng, multi, nf)
            if r is None:
                continue
            solved += 1
            ok += r[0]; cyc += r[1]; badref += r[2]
        out[label] = dict(instances=solved, all_components_valid=ok, with_cycle=cyc,
                          with_wrong_root_count=badref)
    R["V4"] = out
    print("V4", out)


def v5_ev():
    dt = 5 / 60
    ex = []
    for Tmin in (4, 5, 10, 12, 60):
        n_e = math.floor((Tmin / 60) / dt + 1e-9) + 1
        ex.append(dict(T_dep_min=Tmin, n_e=n_e, boundary_time_min=(n_e - 1) * 5,
                       available_intervals=list(range(1, n_e)), ok=(n_e - 1) * 5 <= Tmin))
    net = small_test_net()
    net.loads[2] = Load(20, 6, tier=1, rho=0.5)        # 10-20 kW curtailable load
    # power-limited grid-forming BESS (25 kW): during the 20 kW pick-up interval it must hold
    # 20 kW of headroom for the step (11), so serving load then requires EV discharge, which
    # the departure requirement (10) limits
    net.resources = [Resource("B", 1, 'B', Pmax=25, Pch=25, rated=100, Qmax=60, droop=0.01, E=200,
                              BS=1, GFM=1, FR=1, VR=1, PROT=1, If=2.0, soc0=0.9, socmin=0.2),
                     Resource("EV", 2, 'V', Pmax=11, Pch=11, rated=11, E=60, GFL=1, soc0=0.612,
                              Tdep=10 / 60, socreq=0.6, avail=np.ones(6), socmin=0.2)]
    c = Case(net, H=6, dt=dt, islands=[{"C": {1, 2}, "refs": {"B"}}], oos={1, 2},
             params=dict(Nsw=1, x_dwell=False))
    sol, info, _ = solve(c)
    R["V5"] = dict(examples=ex, solved=dict(status=info["highs_status"],
                   soc_initial=0.612, soc_required=0.6,
                   soc_required_effective=sol["meta"]["ev_info"]["EV"]["req_eff"],
                   soc_at_boundary_3=sol["SOC"]["EV", 3],
                   ev_charge_kW=[round(sol["Pch"]["EV", t] * 1000, 2) for t in range(1, 7)],
                   u_ev=[round(sol["u"]["EV", t]) for t in range(1, 7)],
                   ev_discharge_kW=[round(sol["Pdis"]["EV", t] * 1000, 2) for t in range(1, 7)],
                   served_kW=[round(sol["d"][2, t] * 1000, 2) for t in range(1, 7)]))
    print("V5", R["V5"])


def v6_dwell():
    net = small_test_net()
    # switch opened at tau = 0 (history closed, closed, open); with Nsw = 3 it may not
    # re-close before t = 3
    c = Case(net, H=5, dt=5 / 60, islands=[{"C": {1, 2}, "refs": {"G"}}], oos={1, 2},
             hist_s={0: [1, 1, 1, 0]}, params=dict(Nsw=3, x_dwell=False))
    sol, info, (m, V, meta) = solve(c)
    states = [round(sol["se"][0, t]) for t in range(1, 6)]
    R["V6"] = dict(history=[1, 1, 1, 0], Nsw=3, switch_states_t1_to_t5=states,
                   violations=checks.dwell_violations(sol, meta, 3, {0: [1, 1, 1, 0]}),
                   expected_first_closure=3)
    print("V6", R["V6"])


if __name__ == "__main__":
    v1_identity(); v2_secant(); v3_polygon(); v4_prop1(); v5_ev(); v6_dwell()
    with open(os.path.join(OUT, "verification.json"), "w") as f:
        json.dump(R, f, indent=2, default=float)
    print("saved results/verification.json")
