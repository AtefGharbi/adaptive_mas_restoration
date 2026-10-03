"""Island-level restoration MILP (paper Sections III-IV, VII-D(ii)).

One builder serves both uses in the paper:
  * single-island problem (23) solved by an Island Coordinator: one island, candidate C_k,
    reference set G^ref_k = all eligible references in C_k;
  * multi-island centralized baseline B1: K islands over the whole affected area, each with
    island-indexed topology/connectivity variables (y_{i,k,t}, s_{ij,k,t}, F_{ij,k,t},
    z_{g,k,t}) and island-level constraints (11)-(13), (27) written with membership-linearized
    contributions.  Bus/edge/resource electrical variables are shared (a bus belongs to at most
    one island).

All quantities are per unit on S_base inside the model.
"""
import math
import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pulp

DEFAULTS = dict(
    Vmin=0.95, Vmax=1.05, Mp=12, L=8, Nsw=3, dfmax_frac=0.01, Tres=0.25, Pres=0.0,
    Iprot=0.25, weights=(1 / 6,) * 6, eps=None, sigma_load=0.10, norm_mode="energy", no_conn_flow=False,
    x_dwell=True, bp_spacing="quadratic", time_limit=60, gap=0.01, load_profile=None, open_area_first=True,
)


@dataclass
class Case:
    net: object
    H: int
    dt: float                      # hours
    islands: List[dict]            # each: {"C": set(buses), "refs": set(resource names)}
    oos: Set[int]
    faulted: Set[int] = field(default_factory=set)
    t0: int = 0                    # interval offset into profiles (rolling re-solve)
    hist_s: Dict[int, List[int]] = field(default_factory=dict)   # edge idx -> states tau<=0
    hist_x: Dict[int, List[int]] = field(default_factory=dict)   # bus -> pick-up tau<=0
    soc_init: Dict[str, float] = field(default_factory=dict)
    unavailable: Set[str] = field(default_factory=set)
    norm: Optional[dict] = None
    params: dict = field(default_factory=dict)

    def p(self, key):
        return self.params.get(key, DEFAULTS[key])


# ------------------------------------------------------------------------------------------
def affected_loads(net, oos):
    return sorted(b for b in oos if b in net.loads and net.loads[b].P > 0)


def load_forecast(case, i, t):
    """\\hat P^L_{i,t} in p.u. (t = 1..H)."""
    prof = case.p("load_profile")
    m = 1.0 if prof is None else prof[case.t0 + t - 1]
    return case.net.loads[i].P * m / case.net.S_base


def renewable_forecast(case, g, t):
    return g.rated * g.profile[case.t0 + t - 1] / case.net.S_base


def compute_norm(case):
    """Normalization constants of Section IV, fixed at fault time over the affected area."""
    net, H, dt = case.net, case.H, case.dt
    aff = affected_loads(net, case.oos)
    t1 = [i for i in aff if net.loads[i].tier == 1]
    Eall = sum(load_forecast(case, i, t) * dt for t in range(1, H + 1) for i in aff)
    ET1 = sum(load_forecast(case, i, t) * dt for t in range(1, H + 1) for i in t1)
    W = sum(w_of(net.loads[i].tier) * load_forecast(case, i, t) * dt
            for t in range(1, H + 1) for i in aff)
    sw = [e for e in net.edges if e.switchable and e.i in case.oos and e.j in case.oos
          and e.idx not in case.faulted]
    Nsw = case.p("Nsw")
    Csw = math.ceil(H / Nsw) * len(sw)          # c_ij^sw = 1
    stor = [g for g in net.resources if g.cls in "BV" and g.bus in case.oos]
    Cdeg = sum(g.cdeg * g.Pmax / net.S_base * dt for t in range(1, H + 1) for g in stor)
    if Eall <= 0:
        raise ValueError("affected-load set is empty: objective (19) is not instantiated")
    # Energy-based penalty terms share the base E_all, so that weights are relative values per
    # kWh (Section IV).  Cdeg is kept only for reporting the older, capacity-based normalization.
    if case.p("norm_mode") == "capacity":     # V6 normalization, kept for the ablation
        Lb, Db = 0.10 * Eall, (Cdeg if Cdeg > 0 else Eall)
    else:
        Lb, Db = Eall, Eall
    return dict(Eall=Eall, ET1=ET1 if t1 else None, W=W, Csw=max(Csw, 1),
                L=Lb, Cdeg=Db, aff=aff, t1=t1)


def w_of(tier):
    return {1: 100.0, 2: 10.0, 3: 1.0}[tier]


def secant_breakpoints(smax, L, spacing="quadratic"):
    """Breakpoints on [-smax, smax]; L even so that 0 is a breakpoint (zero loss at zero flow).
    'quadratic' spacing places breakpoints at +/- smax (m/(L/2))^2, dense near zero flow."""
    assert L % 2 == 0, "L must be even so that 0 is a breakpoint (zero loss at zero flow)"
    if spacing == "uniform":
        return np.linspace(-smax, smax, L + 1)
    h = smax * (np.arange(0, L // 2 + 1) / (L // 2)) ** 2
    return np.concatenate([-h[::-1], h[1:]])


def clipped_moments(mean, sd, lo, hi, n=40000, seed=0):
    """Mean and SD of min(max(mean+eps,lo),hi) for Gaussian eps (used for E[P^act])."""
    if sd <= 0:
        v = min(max(mean, lo), hi)
        return v, 0.0
    rng = np.random.default_rng(seed)
    s = np.clip(mean + sd * rng.standard_normal(n), lo, hi)
    return float(s.mean()), float(s.std())


# ------------------------------------------------------------------------------------------
def build(case: Case):
    net, H, dt = case.net, case.H, case.dt
    T = list(range(1, H + 1))
    Sb = net.S_base
    K = list(range(len(case.islands)))
    Cset = set().union(*[isl["C"] for isl in case.islands])
    buses = sorted(Cset)
    norm = case.norm or compute_norm(case)
    aff = [i for i in norm["aff"] if i in Cset]
    t1 = [i for i in norm["t1"] if i in Cset]
    E = [e for e in net.edges if e.i in Cset and e.j in Cset and e.idx not in case.faulted]
    Ek = {k: [e for e in E if e.i in case.islands[k]["C"] and e.j in case.islands[k]["C"]]
          for k in K}
    res = [g for g in net.resources if g.bus in Cset and g.name not in case.unavailable]
    D = [g for g in res if g.cls == 'D']
    R = [g for g in res if g.cls == 'R']
    S = [g for g in res if g.cls in 'BV']
    V = [g for g in res if g.cls == 'V']
    refs = {k: [g for g in res if g.name in case.islands[k]["refs"]
                and g.bus in case.islands[k]["C"] and g.GFM and g.BS] for k in K}
    Vmin, Vmax, Mp, L = case.p("Vmin"), case.p("Vmax"), case.p("Mp"), case.p("L")
    Nsw = case.p("Nsw")
    BIG = 10.0                                      # p.u. big-M for power quantities
    nb = len(buses)
    multi = len(K) > 1
    m = pulp.LpProblem("island_restoration", pulp.LpMaximize)
    LV = pulp.LpVariable
    B = pulp.LpBinary

    # ---------------- topology / energization ----------------
    y = {(i, k, t): LV(f"y_{i}_{k}_{t}", cat=B) for k in K for i in case.islands[k]["C"] for t in T}
    yb = {(k, t): LV(f"yb_{k}_{t}", cat=B) for k in K for t in T}
    yi = {(i, t): pulp.lpSum(y[i, k, t] for k in K if (i, k, t) in y) for i in buses for t in T}
    s = {(e.idx, k, t): LV(f"s_{e.idx}_{k}_{t}", cat=B) for k in K for e in Ek[k] for t in T}
    se = {(e.idx, t): pulp.lpSum(s[e.idx, k, t] for k in K if (e.idx, k, t) in s)
          for e in E for t in T}
    for i in buses:
        for t in T:
            m += yi[i, t] <= 1
    for k in K:
        Ck = case.islands[k]["C"]
        for t in T:
            for i in Ck:
                m += y[i, k, t] <= yb[k, t]                                   # (1b)
            m += yb[k, t] <= pulp.lpSum(y[i, k, t] for i in Ck)               # (1b)
    for e in E:
        for t in T:
            for k in K:
                if (e.idx, k, t) in s:
                    m += s[e.idx, k, t] <= y[e.i, k, t]
                    m += s[e.idx, k, t] <= y[e.j, k, t]
            if not e.switchable:                                            # healthy line
                m += yi[e.i, t] == yi[e.j, t]
                for k in K:
                    a, b_ = (e.i, k, t) in y, (e.j, k, t) in y
                    if a and b_:
                        m += y[e.i, k, t] == y[e.j, k, t]
                        m += s[e.idx, k, t] == y[e.i, k, t]
                    elif a:
                        m += y[e.i, k, t] == 0
                    elif b_:
                        m += y[e.j, k, t] == 0

    # ---------------- reference selection (1a) ----------------
    z = {(g.name, k, t): LV(f"z_{g.name}_{k}_{t}", cat=B) for k in K for g in refs[k] for t in T}
    u = {(g.name, t): LV(f"u_{g.name}_{t}", cat=B) for g in res for t in T}
    for k in K:
        for t in T:
            m += pulp.lpSum(z[g.name, k, t] for g in refs[k]) == yb[k, t]
            for g in refs[k]:
                m += z[g.name, k, t] <= u[g.name, t]
            for i in case.islands[k]["C"]:
                here = [g for g in refs[k] if g.bus == i]
                if here:
                    m += pulp.lpSum(z[g.name, k, t] for g in here) <= y[i, k, t]
    for g in res:
        for t in T:
            m += u[g.name, t] <= yi[g.bus, t]                                # (8b)
            if g.cls == 'V' and g.avail is not None:
                m += u[g.name, t] <= int(g.avail[case.t0 + t - 1])

    # ---------------- connectivity flow (14) and edge count (15) ----------------
    F = {(e.idx, k, t): LV(f"F_{e.idx}_{k}_{t}", -nb, nb) for k in K for e in Ek[k] for t in T}
    Fr = {(g.name, k, t): LV(f"Fr_{g.name}_{k}_{t}", 0, nb) for k in K for g in refs[k] for t in T}
    for k in K:
        Ck = case.islands[k]["C"]
        for t in T:
            for e in Ek[k]:
                m += F[e.idx, k, t] <= nb * s[e.idx, k, t]
                m += -F[e.idx, k, t] <= nb * s[e.idx, k, t]
            for g in refs[k]:
                m += Fr[g.name, k, t] <= nb * z[g.name, k, t]
            for i in (Ck if not case.p("no_conn_flow") else []):
                out_ = pulp.lpSum(F[e.idx, k, t] for e in Ek[k] if e.i == i)
                in_ = pulp.lpSum(F[e.idx, k, t] for e in Ek[k] if e.j == i)
                root = pulp.lpSum(Fr[g.name, k, t] for g in refs[k] if g.bus == i)
                m += out_ - in_ == root - y[i, k, t]                          # (14)
            m += (pulp.lpSum(s[e.idx, k, t] for e in Ek[k])
                  == pulp.lpSum(y[i, k, t] for i in Ck)
                  - pulp.lpSum(z[g.name, k, t] for g in refs[k]))             # (15)

    # ---------------- loads (2) ----------------
    x = {(i, t): LV(f"x_{i}_{t}", cat=B) for i in aff for t in T}
    d = {(i, t): LV(f"d_{i}_{t}", 0) for i in aff for t in T}
    for i in aff:
        Ld = net.loads[i]
        for t in T:
            Ph = load_forecast(case, i, t)
            m += d[i, t] >= (1 - Ld.rho) * Ph * x[i, t]
            m += d[i, t] <= Ph * x[i, t]
            m += x[i, t] <= yi[i, t]
    tanphi = {i: (net.loads[i].Q / net.loads[i].P) for i in aff}

    # ---------------- DG dispatch (8), (8a) ----------------
    P = {(g.name, t): LV(f"P_{g.name}_{t}", 0) for g in D + R for t in T}
    Q = {(g.name, t): LV(f"Q_{g.name}_{t}", -g.Qmax / Sb, g.Qmax / Sb) for g in res for t in T}
    for g in D:
        for t in T:
            m += P[g.name, t] >= g.Pmin / Sb * u[g.name, t]
            m += P[g.name, t] <= g.Pmax / Sb * u[g.name, t]
    for g in R:
        for t in T:
            m += P[g.name, t] <= renewable_forecast(case, g, t) * u[g.name, t]
    for g in res:      # EVs have VR = 0 (unity power factor); BESS may regulate voltage
        for t in T:
            m += Q[g.name, t] <= g.VR * g.Qmax / Sb * u[g.name, t]
            m += -Q[g.name, t] <= g.VR * g.Qmax / Sb * u[g.name, t]

    # ---------------- storage (8b), (9), (10) ----------------
    Pch = {(g.name, t): LV(f"Pch_{g.name}_{t}", 0) for g in S for t in T}
    Pdis = {(g.name, t): LV(f"Pdis_{g.name}_{t}", 0) for g in S for t in T}
    chi = {(g.name, t): LV(f"chi_{g.name}_{t}", cat=B) for g in S for t in T}
    SOC = {(g.name, t): LV(f"SOC_{g.name}_{t}", g.socmin, g.socmax) for g in S for t in range(1, H + 2)}
    ev_info = {}
    for g in S:
        soc1 = case.soc_init.get(g.name, g.soc0)
        m += SOC[g.name, 1] == soc1
        E_pu = g.E / Sb
        for t in T:
            m += Pch[g.name, t] <= g.Pch / Sb * chi[g.name, t]
            m += Pdis[g.name, t] <= g.Pmax / Sb * (u[g.name, t] - chi[g.name, t])
            m += chi[g.name, t] <= u[g.name, t]
            m += SOC[g.name, t + 1] == (SOC[g.name, t] + g.eta_c * Pch[g.name, t] * dt / E_pu
                                        - Pdis[g.name, t] * dt / (g.eta_d * E_pu))      # (9)
        if g.cls == 'V':
            Tdep = g.Tdep - case.t0 * dt                 # departure after this solve instant
            if Tdep <= 0:
                for t in T:
                    m += u[g.name, t] == 0
                continue
            n_e = math.floor(Tdep / dt + 1e-9) + 1
            for t in T:
                if t >= n_e:
                    m += u[g.name, t] == 0                                  # (10)
            # ESA commitment: the island may not leave the vehicle with less than
            # min(SOC_req, SOC at the solve instant) at departure.  Charging is optional load,
            # so (10) can never make the island problem infeasible.
            req_eff = min(g.socreq, soc1)
            ev_info[g.name] = dict(n_e=n_e, req_eff=req_eff)
            if n_e <= H + 1:
                m += SOC[g.name, n_e] >= req_eff                              # (10)
            else:
                m += SOC[g.name, H + 1] >= req_eff                            # (10), beyond horizon

    # ---------------- network: flows, voltages, ratings, losses (3)-(7a) ----------------
    Pf = {(e.idx, t): LV(f"Pf_{e.idx}_{t}") for e in E for t in T}
    Qf = {(e.idx, t): LV(f"Qf_{e.idx}_{t}") for e in E for t in T}
    v = {(i, t): LV(f"v_{i}_{t}", 0, Vmax ** 2) for i in buses for t in T}
    ell = {(e.idx, t): LV(f"ell_{e.idx}_{t}", 0) for e in E for t in T}
    psP = {(e.idx, t): LV(f"psP_{e.idx}_{t}", 0) for e in E for t in T}
    psQ = {(e.idx, t): LV(f"psQ_{e.idx}_{t}", 0) for e in E for t in T}
    cosf = math.cos(math.pi / Mp)
    for e in E:
        smax = e.smax / Sb
        bp = secant_breakpoints(smax, L, case.p("bp_spacing"))
        Me = Vmax ** 2 + 2 * math.hypot(e.r, e.x) * smax
        for t in T:
            for mm in range(1, Mp + 1):
                th = 2 * math.pi * mm / Mp
                m += Pf[e.idx, t] * math.cos(th) + Qf[e.idx, t] * math.sin(th) <= cosf * smax * se[e.idx, t]  # (7)
            dv = v[e.i, t] - v[e.j, t] - 2 * (e.r * Pf[e.idx, t] + e.x * Qf[e.idx, t])
            m += dv <= Me * (1 - se[e.idx, t])                                 # (5)
            m += -dv <= Me * (1 - se[e.idx, t])
            for a_, b_ in zip(bp[:-1], bp[1:]):                                # (7a)
                m += psP[e.idx, t] >= (a_ + b_) * Pf[e.idx, t] - a_ * b_
                m += psQ[e.idx, t] >= (a_ + b_) * Qf[e.idx, t] - a_ * b_
            m += ell[e.idx, t] >= e.r * (psP[e.idx, t] + psQ[e.idx, t])
    for i in buses:
        for t in T:
            m += v[i, t] >= Vmin ** 2 * yi[i, t]                               # (6)
            m += v[i, t] <= Vmax ** 2 * yi[i, t]
            inj = (pulp.lpSum(P[g.name, t] for g in D + R if g.bus == i)
                   + pulp.lpSum(Pdis[g.name, t] - Pch[g.name, t] for g in S if g.bus == i)
                   - (d[i, t] if i in aff else 0)
                   - 0.5 * pulp.lpSum(ell[e.idx, t] for e in E if i in (e.i, e.j)))
            qinj = (pulp.lpSum(Q[g.name, t] for g in res if g.bus == i)
                    - (tanphi[i] * d[i, t] if i in aff else 0))
            out_P = pulp.lpSum(Pf[e.idx, t] for e in E if e.i == i)
            in_P = pulp.lpSum(Pf[e.idx, t] for e in E if e.j == i)
            out_Q = pulp.lpSum(Qf[e.idx, t] for e in E if e.i == i)
            in_Q = pulp.lpSum(Qf[e.idx, t] for e in E if e.j == i)
            m += inj == out_P - in_P, f"Pbal_{i}_{t}"                         # (3)
            m += qinj == out_Q - in_Q, f"Qbal_{i}_{t}"                        # (4)

    # ---------------- membership helpers for island-level constraints ----------------
    def member_ub(expr_ub_val, var, bus, k, t, name):
        """Variable w <= var, w <= ub * y_{bus,k,t} (upper bound of k-contribution)."""
        if not multi:
            return var
        w = LV(name, 0)
        m.__iadd__(w <= var)
        m.__iadd__(w <= expr_ub_val * y[bus, k, t] if (bus, k, t) in y else w <= 0)
        return w

    def member_lb(var, ub, bus, k, t, name):
        """Variable w >= var - ub*(1 - y_{bus,k,t}), w >= 0 (lower bound of k-contribution)."""
        if not multi:
            return var
        w = LV(name, 0)
        if (bus, k, t) in y:
            m.__iadd__(w >= var - ub * (1 - y[bus, k, t]))
        return w

    # ---------------- headroom (10a), droop screen (11), reserve split (12) ----------------
    dfr = case.p("dfmax_frac")
    Tres = case.p("Tres")
    hFR = {(g.name, t): LV(f"hFR_{g.name}_{t}", 0) for g in res for t in T}
    hCR = {(g.name, t): LV(f"hCR_{g.name}_{t}", 0) for g in res for t in T}
    phi = {(g.name, t): LV(f"phi_{g.name}_{t}", 0) for g in res for t in T}
    socres = {g.name: (max(g.socmin, g.socreq) if g.cls == 'V' else g.socmin) for g in S}
    for g in res:
        for t in T:
            hsum = hFR[g.name, t] + hCR[g.name, t]
            if g.cls == 'D':
                head = [g.Pmax / Sb * u[g.name, t] - P[g.name, t]]
            elif g.cls == 'R':
                head = [renewable_forecast(case, g, t) * u[g.name, t] - P[g.name, t]]
            else:   # converter bound (includes ceasing to charge) and energy bound
                head = [g.Pmax / Sb * u[g.name, t] - Pdis[g.name, t] + Pch[g.name, t],
                        g.eta_d * (g.E / Sb) * (SOC[g.name, t] - socres[g.name]) / Tres]
            for hb in head:
                m += hsum <= hb                                               # (12)
                m += hFR[g.name, t] <= g.FR * hb if g.FR else hFR[g.name, t] <= 0
            m += phi[g.name, t] <= g.FR * dfr * (g.rated / Sb) / g.droop * u[g.name, t]   # (11)
            m += phi[g.name, t] <= hFR[g.name, t]
    dP = {(k, t): LV(f"dPstep_{k}_{t}", 0) for k in K for t in T}
    for k in K:
        Ck = case.islands[k]["C"]
        for t in T:
            m += pulp.lpSum(member_ub(BIG, phi[g.name, t], g.bus, k, t, f"phik_{g.name}_{k}_{t}")
                            for g in res if g.bus in Ck) >= dP[k, t]         # (11)
            m += pulp.lpSum(member_ub(BIG, hCR[g.name, t], g.bus, k, t, f"hCRk_{g.name}_{k}_{t}")
                            for g in res if g.bus in Ck) >= case.p("Pres") * yb[k, t]   # (12)
            # credible step: largest non-reference online injection, largest pick-up block
            for g in res:
                if g.bus not in Ck:
                    continue
                zg = z[g.name, k, t] if (g.name, k, t) in z else 0
                out = P[g.name, t] if g.cls in 'DR' else Pdis[g.name, t]
                m += dP[k, t] >= out - BIG * (1 - y[g.bus, k, t]) - BIG * zg
            for i in aff:
                if i not in Ck:
                    continue
                xprev = x[i, t - 1] if t > 1 else (case.hist_x.get(i, [0])[-1])
                m += dP[k, t] >= load_forecast(case, i, t) * (x[i, t] - xprev) - BIG * (1 - y[i, k, t])
            # heuristic protection screen (13)
            m += pulp.lpSum(g.PROT * g.If * member_ub(1, u[g.name, t], g.bus, k, t, f"uk_{g.name}_{k}_{t}")
                            for g in res if g.bus in Ck and g.PROT) >= case.p("Iprot") * yb[k, t]

    # ---------------- dwell (16) with history ----------------
    om = {}
    sw_edges = [e for e in E if e.switchable]
    for e in sw_edges:
        hist = case.hist_s.get(e.idx, [0] * Nsw)
        prev = hist[-1]
        for t in T:
            om[e.idx, t] = LV(f"om_{e.idx}_{t}", 0, 1)
            cur = se[e.idx, t]
            pv = se[e.idx, t - 1] if t > 1 else prev
            m += om[e.idx, t] >= cur - pv
            m += om[e.idx, t] >= pv - cur
        for t in T:
            window = []
            hist_changes = [abs(hist[a] - hist[a - 1]) for a in range(1, len(hist))]
            for tau in range(t - Nsw + 1, t + 1):
                if tau >= 1:
                    window.append(om[e.idx, tau])
                else:   # tau <= 0: logged change between boundaries tau-1 and tau
                    idx = len(hist_changes) + tau - 1
                    if idx >= 0:
                        window.append(hist_changes[idx])
            m += pulp.lpSum(window) <= 1                                      # (16)
    omx = {}
    if case.p("x_dwell"):
        for i in aff:
            hist = case.hist_x.get(i, [0] * Nsw)
            for t in T:
                omx[i, t] = LV(f"omx_{i}_{t}", 0, 1)
                pv = x[i, t - 1] if t > 1 else hist[-1]
                m += omx[i, t] >= x[i, t] - pv
                m += omx[i, t] >= pv - x[i, t]
            hist_changes = [abs(hist[a] - hist[a - 1]) for a in range(1, len(hist))]
            for t in T:
                window = []
                for tau in range(t - Nsw + 1, t + 1):
                    if tau >= 1:
                        window.append(omx[i, tau])
                    else:
                        idx = len(hist_changes) + tau - 1
                        if idx >= 0:
                            window.append(hist_changes[idx])
                m += pulp.lpSum(window) <= 1

    # ---------------- chance constraint (27) ----------------
    eps = case.p("eps")
    Sav = {}
    if eps is not None:
        kap = math.sqrt((1 - eps) / eps)
        sl = case.p("sigma_load")
        for g in S:
            for t in T:
                Sav[g.name, t] = LV(f"Sav_{g.name}_{t}", 0)
                m += Sav[g.name, t] <= g.Pmax / Sb * u[g.name, t]
                m += Sav[g.name, t] <= g.eta_d * (g.E / Sb) * (SOC[g.name, t] - socres[g.name]) / Tres
        mom = {}
        for g in R:
            for t in T:
                f = renewable_forecast(case, g, t)
                mom[g.name, t] = clipped_moments(f, g.sigma * g.rated / Sb, 0.0, g.rated / Sb,
                                                 seed=zlib.crc32(f"{g.name}|{case.t0 + t}".encode()))
        for k in K:
            Ck = case.islands[k]["C"]
            for t in T:
                Pl_k = pulp.lpSum(member_lb(ell[e.idx, t], BIG, e.i, k, t, f"ellk_{e.idx}_{k}_{t}")
                                  for e in Ek[k])
                dem = pulp.lpSum(member_lb(d[i, t], BIG, i, k, t, f"dk_{i}_{k}_{t}") for i in aff if i in Ck)
                chg = pulp.lpSum(member_lb(Pch[g.name, t], BIG, g.bus, k, t, f"chk_{g.name}_{k}_{t}")
                                 for g in S if g.bus in Ck)
                supD = pulp.lpSum(g.Pmax / Sb * member_ub(1, u[g.name, t], g.bus, k, t, f"udk_{g.name}_{k}_{t}")
                                  for g in D if g.bus in Ck)
                supR = pulp.lpSum(mom[g.name, t][0] * member_ub(1, u[g.name, t], g.bus, k, t, f"urk_{g.name}_{k}_{t}")
                                  for g in R if g.bus in Ck)
                supS = pulp.lpSum((g.pi if g.cls == 'V' else 1.0) *
                                  member_ub(BIG, Sav[g.name, t], g.bus, k, t, f"sk_{g.name}_{k}_{t}")
                                  for g in S if g.bus in Ck)
                mu = dem + chg + Pl_k + dP[k, t] - supD - supR - supS
                sig = (pulp.lpSum(sl * load_forecast(case, i, t) *
                                  member_lb(x[i, t], 1, i, k, t, f"xk_{i}_{k}_{t}") for i in aff if i in Ck)
                       + pulp.lpSum(mom[g.name, t][1] *
                                    member_lb(u[g.name, t], 1, g.bus, k, t, f"urlk_{g.name}_{k}_{t}")
                                    for g in R if g.bus in Ck)
                       + pulp.lpSum(math.sqrt(g.pi * (1 - g.pi)) *
                                    member_lb(Sav[g.name, t], BIG, g.bus, k, t, f"svk_{g.name}_{k}_{t}")
                                    for g in V if g.bus in Ck))
                m += mu + kap * sig <= 0, f"cc_{k}_{t}"                       # (27)

    # ---------------- objective (19)-(22) ----------------
    a, b, c, dl, et, ze = case.p("weights")
    J = {}
    J["res"] = pulp.lpSum(w_of(net.loads[i].tier) * d[i, t] * dt for i in aff for t in T) / norm["W"]
    J["crit"] = (pulp.lpSum(d[i, t] * dt for i in t1 for t in T) / norm["ET1"]) if norm["ET1"] else 0
    J["sw"] = pulp.lpSum(om.values()) / norm["Csw"]
    J["loss"] = pulp.lpSum(ell[e.idx, t] * dt for e in E for t in T) / norm["L"]
    J["ens"] = pulp.lpSum((load_forecast(case, i, t) - d[i, t]) * dt for i in aff for t in T) / norm["Eall"]
    J["deg"] = pulp.lpSum(g.cdeg * Pdis[g.name, t] * dt for g in S for t in T) / norm["Cdeg"]
    if not norm["ET1"]:
        a = 0.0
    m += a * J["crit"] + b * J["res"] - c * J["sw"] - dl * J["loss"] - et * J["ens"] - ze * J["deg"]

    vars_ = dict(y=y, yb=yb, yi=yi, s=s, se=se, z=z, u=u, F=F, Fr=Fr, x=x, d=d, P=P, Q=Q, Pch=Pch,
                 Pdis=Pdis, chi=chi, SOC=SOC, Pf=Pf, Qf=Qf, v=v, ell=ell, psP=psP, psQ=psQ,
                 hFR=hFR, hCR=hCR, phi=phi, dP=dP, om=om, omx=omx, J=J, Sav=Sav)
    meta = dict(T=T, K=K, buses=buses, E=E, Ek=Ek, res=res, D=D, R=R, S=S, V=V, aff=aff, t1=t1,
                refs=refs, norm=norm, tanphi=tanphi, dt=dt, ev_info=ev_info, nvars=len(m.variables()),
                ncons=len(m.constraints))
    return m, vars_, meta


def val(x):
    if isinstance(x, (int, float)):
        return float(x)
    v = pulp.value(x)
    return 0.0 if v is None else float(v)


def solve(case: Case, verbose=False):
    t0 = time.time()
    m, V, meta = build(case)
    tb = time.time() - t0
    solver = pulp.HiGHS(msg=verbose, timeLimit=case.p("time_limit"), gapRel=case.p("gap"))
    t1 = time.time()
    m.solve(solver)
    ts = time.time() - t1
    status = pulp.LpStatus[m.status]
    sol = extract(case, m, V, meta) if status in ("Optimal",) or m.status == 1 else None
    if sol is None and pulp.value(m.objective) is not None:
        sol = extract(case, m, V, meta)
    h = getattr(m, "solverModel", None)
    hs, gap = status, None
    if h is not None:
        try:
            hs = h.modelStatusToString(h.getModelStatus())
            gap = float(h.getInfo().mip_gap)
        except Exception:
            pass
    info = dict(status=status, highs_status=hs, mip_gap=gap, build_s=tb, solve_s=ts,
                nvars=meta["nvars"], ncons=meta["ncons"],
                obj=val(m.objective) if pulp.value(m.objective) is not None else None)
    return sol, info, (m, V, meta)


def extract(case, m, V, meta):
    T = meta["T"]
    r = lambda d: {k: val(v) for k, v in d.items()}
    sol = {k: r(V[k]) for k in ("y", "yb", "s", "z", "u", "x", "d", "P", "Q", "Pch", "Pdis", "chi",
                                "SOC", "Pf", "Qf", "v", "ell", "psP", "psQ", "hFR", "hCR", "phi",
                                "dP", "om", "omx", "F", "Fr", "Sav")}
    sol["yi"] = {k: val(v) for k, v in V["yi"].items()}
    sol["se"] = {k: val(v) for k, v in V["se"].items()}
    sol["J"] = {k: val(v) for k, v in V["J"].items()}
    sol["meta"] = meta
    return sol
