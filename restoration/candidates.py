"""Stage 3 helpers: sections, candidate islands, and the adequacy screen (18)."""
import networkx as nx
from .model import load_forecast


def sections(net, oos, faulted):
    """Components of the OOS area joined by healthy NON-switchable lines."""
    G = nx.Graph()
    G.add_nodes_from(oos)
    for e in net.edges:
        if e.idx in faulted or e.switchable or e.i not in oos or e.j not in oos:
            continue
        G.add_edge(e.i, e.j)
    secs = [frozenset(c) for c in nx.connected_components(G)]
    of = {b: s for s in secs for b in s}
    SG = nx.Graph()
    SG.add_nodes_from(secs)
    for e in net.edges:
        if e.idx in faulted or not e.switchable or e.i not in oos or e.j not in oos:
            continue
        if of[e.i] != of[e.j]:
            SG.add_edge(of[e.i], of[e.j])
    return secs, of, SG


def eligible_refs(net, oos, unavailable=()):
    return [g for g in net.resources if g.bus in oos and g.GFM and g.BS and g.name not in unavailable]


def nearest_reference_partition(net, oos, faulted, unavailable=()):
    """Assign every section to the nearest eligible reference (hop distance on the section
    graph; ties broken by resource order).  Returns list of {"C", "refs"} candidates."""
    secs, of, SG = sections(net, oos, faulted)
    refs = eligible_refs(net, oos, unavailable)
    best = {}
    for r_idx, g in enumerate(refs):
        dist = nx.single_source_shortest_path_length(SG, of[g.bus])
        for sec, dd in dist.items():
            if sec not in best or (dd, r_idx) < best[sec][:2]:
                best[sec] = (dd, r_idx, g)
    groups = {}
    for sec, (_, r_idx, g) in best.items():
        groups.setdefault(r_idx, set()).update(sec)
    out = []
    for r_idx, C in sorted(groups.items()):
        out.append({"C": set(C), "refs": {g.name for g in refs if g.bus in C}})
    return out


def bfs_shells(net, oos, faulted, seed_bus):
    """Nested connected candidates (distance shells on the section graph) from a seed bus."""
    secs, of, SG = sections(net, oos, faulted)
    dist = nx.single_source_shortest_path_length(SG, of[seed_bus])
    shells, C = [], set()
    for dd in range(max(dist.values()) + 1):
        for sec, d_ in sorted(dist.items(), key=lambda kv: min(kv[0])):
            if d_ == dd:
                C |= set(sec)
        shells.append(set(C))
    return shells


def adequacy_screen(case, C, unavailable=()):
    """Necessary condition (18) for full pick-up of C at minimum curtailment, every t."""
    net, Sb, dt = case.net, case.net.S_base, case.dt
    res = [g for g in net.resources if g.bus in C and g.name not in unavailable]
    for t in range(1, case.H + 1):
        lhs = sum((1 - net.loads[i].rho) * load_forecast(case, i, t)
                  for i in C if i in net.loads and net.loads[i].P > 0)
        rhs = 0.0
        for g in res:
            if g.cls == 'D':
                rhs += g.Pmax / Sb
            elif g.cls == 'R':
                rhs += g.rated * g.profile[case.t0 + t - 1] / Sb
            else:
                if g.cls == 'V' and (g.avail is not None and not g.avail[case.t0 + t - 1]):
                    continue
                energy = g.eta_d * (g.E / Sb) * (g.soc0 - g.socmin) / (t * dt)
                rhs += min(g.Pmax / Sb, energy)
        if lhs > rhs + 1e-9:
            return False
    return True
