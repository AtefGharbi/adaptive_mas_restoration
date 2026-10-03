"""Post-solve validation (paper Sections III-G, VI-C) and algebraic identity checks."""
import math
import networkx as nx
import numpy as np

TOL = 1e-6


def energized_graph(sol, meta, t):
    G = nx.Graph()
    for i in meta["buses"]:
        if sol["yi"][i, t] > 0.5:
            G.add_node(i)
    for e in meta["E"]:
        if sol["se"][e.idx, t] > 0.5:
            G.add_edge(e.i, e.j, e=e)
    return G


def reference_buses(sol, meta, t):
    """Map bus -> list of reference resources with z = 1 at t."""
    out = {}
    for (name, k, tt), val in sol["z"].items():
        if tt == t and val > 0.5:
            g = next(r for r in meta["res"] if r.name == name)
            out.setdefault(g.bus, []).append(g)
    return out


def graph_check(sol, meta):
    """Each energized component must be a tree with exactly one reference (Prop. 1)."""
    rows = []
    for t in meta["T"]:
        G = energized_graph(sol, meta, t)
        refs = reference_buses(sol, meta, t)
        for comp in nx.connected_components(G):
            H = G.subgraph(comp)
            nref = sum(len(refs.get(b, [])) for b in comp)
            rows.append(dict(t=t, n=len(comp), tree=nx.is_tree(H), nref=nref,
                             ok=nx.is_tree(H) and nref == 1))
    return rows


def island_balance_residual(sol, meta, t):
    """Residual of (17) for every energized component (should be ~0 by construction)."""
    G = energized_graph(sol, meta, t)
    out = []
    for comp in nx.connected_components(G):
        dem = sum(sol["d"].get((i, t), 0.0) for i in comp)
        ch = sum(sol["Pch"][g.name, t] for g in meta["S"] if g.bus in comp)
        dis = sum(sol["Pdis"][g.name, t] for g in meta["S"] if g.bus in comp)
        gen = sum(sol["P"][g.name, t] for g in meta["D"] + meta["R"] if g.bus in comp)
        loss = sum(sol["ell"][e.idx, t] for e in meta["E"] if e.i in comp and e.j in comp)
        out.append(dem + ch + loss - gen - dis)
    return out


def loss_overestimate(sol, meta, L, Sb=1000.0):
    """ell >= r(P^2+Q^2) and the excess <= r * 2 * (segment width)^2 / 4 (per-term bound)."""
    worst_under, worst_over_ratio = 0.0, 0.0
    for e in meta["E"]:
        w = 2 * e.smax / Sb / L
        for t in meta["T"]:
            Pf, Qf = sol["Pf"][e.idx, t], sol["Qf"][e.idx, t]
            true = e.r * (Pf ** 2 + Qf ** 2)
            ell = sol["ell"][e.idx, t]
            worst_under = max(worst_under, true - ell)
            bound = e.r * 2 * w ** 2 / 4
            if bound > 0:
                worst_over_ratio = max(worst_over_ratio, (ell - true) / bound)
    return worst_under, worst_over_ratio


def dwell_violations(sol, meta, Nsw, hist_s):
    bad = 0
    for e in meta["E"]:
        if not e.switchable:
            continue
        hist = hist_s.get(e.idx, [0] * Nsw)
        seq = list(hist) + [round(sol["se"][e.idx, t]) for t in meta["T"]]
        ch = [abs(seq[a] - seq[a - 1]) for a in range(1, len(seq))]
        for a in range(len(ch)):
            if sum(ch[max(0, a - Nsw + 1):a + 1]) > 1:
                bad += 1
    return bad


# ------------------------------------------------------------------------------------------
def _sweep(G, comp, root, Snet, V0, tol=1e-10, itmax=200):
    tree = nx.bfs_tree(G.subgraph(comp), root)
    order = list(nx.topological_sort(tree))
    parent = {c: p for p, c in tree.edges()}
    V = {i: complex(V0) for i in comp}
    Ibr = {}
    for _ in range(itmax):
        Isub = {i: np.conj(Snet[i] / V[i]) for i in comp}
        for c in reversed(order):
            if c in parent:
                Ibr[c] = Isub[c]
                Isub[parent[c]] += Isub[c]
        Vold = dict(V)
        for c in order:
            if c in parent:
                e = G[parent[c]][c]["e"]
                V[c] = V[parent[c]] - complex(e.r, e.x) * Ibr[c]
        if max(abs(V[i] - Vold[i]) for i in comp) < tol:
            break
    Sref = Snet[root] + sum(V[root] * np.conj(Ibr[c]) for c in Ibr if parent[c] == root)
    loss = sum(G[parent[c]][c]["e"].r * abs(Ibr[c]) ** 2 for c in Ibr)
    return V, Ibr, parent, Sref, loss


def ac_validate(sol, meta, net, Vmin=0.95, Vmax=1.05, redispatch=True, shed=True, max_iter=6):
    """AC validation policy (Section VI-C).

    Topology and reference assignment are kept as solved.  The reference resource is the slack.
    (1) If its AC output violates its active/reactive limits, the mismatch is re-dispatched to
        the other online units of the same component within their limits (renewables only
        downward, i.e., by curtailment).
    (2) If voltage, loading, or reference limits are still violated, the lowest-priority picked
        load (tier 3 before 2 before 1, ties by largest bus ID) is shed and step (1) repeated.
    Returns aggregate statistics and d_eval (served load after validation, p.u.).
    """
    Sb = net.S_base
    rep = dict(feasible=True, vmin=9.0, vmax=0.0, max_loading=0.0, ac_loss=0.0, model_loss=0.0,
               n_islands=0, n_redispatch=0, n_shed=0, shed_kWh=0.0, failures=[])
    d_eval = dict(sol["d"])
    dt = meta.get("dt", 5 / 60)
    for t in meta["T"]:
        G = energized_graph(sol, meta, t)
        refs = reference_buses(sol, meta, t)
        for comp in nx.connected_components(G):
            rep["n_islands"] += 1
            root = next(b for b in comp if b in refs)
            ref = refs[root][0]
            units = [g for g in meta["res"] if g.bus in comp and g is not ref
                     and sol["u"].get((g.name, t), 0) > 0.5]
            Pset = {g.name: (sol["P"][g.name, t] if g.cls in "DR"
                             else sol["Pdis"][g.name, t] - sol["Pch"][g.name, t]) for g in units}
            Qset = {g.name: sol["Q"].get((g.name, t), 0.0) for g in units}
            picked = [i for i in comp if d_eval.get((i, t), 0) > 1e-9]
            lo, hi = ((ref.Pmin / Sb, ref.Pmax / Sb) if ref.cls == "D" else (-ref.Pch / Sb, ref.Pmax / Sb))
            qmax = ref.Qmax / Sb
            V0 = math.sqrt(max(sol["v"][root, t], 1e-6))
            ok = False
            while True:
                for _ in range(max_iter):
                    Snet = {}
                    for i in comp:
                        Pd = d_eval.get((i, t), 0.0)
                        Snet[i] = complex(Pd, Pd * meta["tanphi"].get(i, 0.0))
                    for g in units:
                        Snet[g.bus] -= complex(Pset[g.name], Qset[g.name])
                    V, Ibr, parent, Sref, loss = _sweep(G, comp, root, Snet, V0)
                    dP = (Sref.real - hi) if Sref.real > hi else ((Sref.real - lo) if Sref.real < lo else 0.0)
                    dQ = (Sref.imag - qmax) if Sref.imag > qmax else ((Sref.imag + qmax) if Sref.imag < -qmax else 0.0)
                    if (abs(dP) < 1e-7 and abs(dQ) < 1e-7) or not redispatch:
                        break
                    rep["n_redispatch"] += 1
                    for g in units:          # shift active-power mismatch to other units
                        if abs(dP) < 1e-9:
                            break
                        if g.cls == "D":
                            pl, ph = g.Pmin / Sb, g.Pmax / Sb
                        elif g.cls == "R":
                            pl, ph = 0.0, Pset[g.name]          # curtail only
                        else:
                            pl, ph = -g.Pch / Sb, g.Pmax / Sb
                        new = min(max(Pset[g.name] + dP, pl), ph)
                        dP -= new - Pset[g.name]
                        Pset[g.name] = new
                    for g in units:          # reactive mismatch to VR units
                        if abs(dQ) < 1e-9 or not g.VR:
                            continue
                        new = min(max(Qset[g.name] + dQ, -g.Qmax / Sb), g.Qmax / Sb)
                        dQ -= new - Qset[g.name]
                        Qset[g.name] = new
                vm = [abs(V[i]) for i in comp]
                loading = [abs(V[parent[c]] * np.conj(Ibr[c])) / (G[parent[c]][c]["e"].smax / Sb) for c in Ibr]
                ref_ok = (lo - 1e-6 <= Sref.real <= hi + 1e-6) and abs(Sref.imag) <= qmax + 1e-6
                v_ok = min(vm) >= Vmin - 1e-4 and max(vm) <= Vmax + 1e-4
                l_ok = (max(loading) if loading else 0) <= 1 + 1e-6
                if ref_ok and v_ok and l_ok:
                    ok = True
                    break
                if not shed or not picked:
                    break
                victim = max(picked, key=lambda i: (net.loads[i].tier, i))
                rep["n_shed"] += 1
                rep["shed_kWh"] += d_eval[victim, t] * Sb * dt
                d_eval[victim, t] = 0.0
                picked.remove(victim)
            if not ok:
                rep["feasible"] = False
                rep["failures"].append((t, ref.name))
            rep["vmin"] = min(rep["vmin"], min(abs(V[i]) for i in comp))
            rep["vmax"] = max(rep["vmax"], max(abs(V[i]) for i in comp))
            if Ibr:
                rep["max_loading"] = max(rep["max_loading"], max(
                    abs(V[parent[c]] * np.conj(Ibr[c])) / (G[parent[c]][c]["e"].smax / Sb) for c in Ibr))
            rep["ac_loss"] += loss * Sb * dt
            rep["model_loss"] += sum(sol["ell"][G[parent[c]][c]["e"].idx, t] for c in Ibr) * Sb * dt
    rep["d_eval"] = d_eval
    return rep
