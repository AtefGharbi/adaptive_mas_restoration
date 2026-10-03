"""Network, resource, and case data for the adaptive MAS service-restoration model.

All powers in the data structures are in kW / kvar / kWh; the model converts them to
per unit on Network.S_base (kVA). Impedances are stored in per unit.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Edge:
    idx: int
    i: int
    j: int
    r: float            # p.u.
    x: float            # p.u.
    smax: float         # kVA
    switchable: bool
    normally_open: bool = False


@dataclass
class Load:
    P: float            # kW (nominal)
    Q: float            # kvar (nominal)
    tier: int = 3
    rho: float = 0.0    # curtailable fraction


@dataclass
class Resource:
    name: str
    bus: int
    cls: str            # 'D', 'R', 'B', 'V'
    Pmax: float = 0.0   # kW (D: max output; R: rated; B/V: discharge limit)
    Pmin: float = 0.0
    Qmax: float = 0.0
    rated: float = 0.0
    droop: float = 0.05
    BS: int = 0
    GFM: int = 0
    GFL: int = 0
    FR: int = 0
    VR: int = 0
    PROT: int = 0
    If: float = 0.0     # fault-current contribution, p.u. current on S_base
    profile: Optional[np.ndarray] = None   # R: forecast availability fraction of rated per interval
    sigma: float = 0.0                     # R: forecast-error SD as fraction of rated
    E: float = 0.0      # kWh
    Pch: float = 0.0    # kW
    eta_c: float = 0.95
    eta_d: float = 0.95
    soc0: float = 0.5
    socmin: float = 0.2
    socmax: float = 0.95
    avail: Optional[np.ndarray] = None     # V: plug-in/enrolled indicator per interval
    Tdep: float = 1e9   # V: departure time after solve instant (h)
    socreq: float = 0.0
    pi: float = 1.0     # V: participation probability
    recharge_credit: bool = False          # V: guaranteed charging access after horizon
    cdeg: float = 0.05  # degradation cost per kWh discharged (dimensionless relative)


@dataclass
class Network:
    name: str
    buses: List[int]
    edges: List[Edge]
    loads: Dict[int, Load]
    resources: List[Resource]
    S_base: float = 1000.0     # kVA
    V_base: float = 12.66      # kV

    def edge_by_ends(self, i, j):
        for e in self.edges:
            if (e.i, e.j) in ((i, j), (j, i)):
                return e
        raise KeyError((i, j))


# --------------------------------------------------------------------------------------
# IEEE 33-bus feeder (Baran & Wu, 1989): (from, to, R ohm, X ohm, P_to kW, Q_to kvar)
# --------------------------------------------------------------------------------------
_IEEE33 = [
    (1, 2, 0.0922, 0.0470, 100, 60), (2, 3, 0.4930, 0.2511, 90, 40),
    (3, 4, 0.3660, 0.1864, 120, 80), (4, 5, 0.3811, 0.1941, 60, 30),
    (5, 6, 0.8190, 0.7070, 60, 20), (6, 7, 0.1872, 0.6188, 200, 100),
    (7, 8, 0.7114, 0.2351, 200, 100), (8, 9, 1.0300, 0.7400, 60, 20),
    (9, 10, 1.0440, 0.7400, 60, 20), (10, 11, 0.1966, 0.0650, 45, 30),
    (11, 12, 0.3744, 0.1238, 60, 35), (12, 13, 1.4680, 1.1550, 60, 35),
    (13, 14, 0.5416, 0.7129, 120, 80), (14, 15, 0.5910, 0.5260, 60, 10),
    (15, 16, 0.7463, 0.5450, 60, 20), (16, 17, 1.2890, 1.7210, 60, 20),
    (17, 18, 0.7320, 0.5740, 90, 40), (2, 19, 0.1640, 0.1565, 90, 40),
    (19, 20, 1.5042, 1.3554, 90, 40), (20, 21, 0.4095, 0.4784, 90, 40),
    (21, 22, 0.7089, 0.9373, 90, 40), (3, 23, 0.4512, 0.3083, 90, 50),
    (23, 24, 0.8980, 0.7091, 420, 200), (24, 25, 0.8960, 0.7011, 420, 200),
    (6, 26, 0.2030, 0.1034, 60, 25), (26, 27, 0.2842, 0.1447, 60, 25),
    (27, 28, 1.0590, 0.9337, 60, 20), (28, 29, 0.8042, 0.7006, 120, 70),
    (29, 30, 0.5075, 0.2585, 200, 600), (30, 31, 0.9744, 0.9630, 150, 70),
    (31, 32, 0.3105, 0.3619, 210, 100), (32, 33, 0.3410, 0.5302, 60, 40),
]
_IEEE33_TIES = [(8, 21, 2.0, 2.0), (9, 15, 2.0, 2.0), (12, 22, 2.0, 2.0),
                (18, 33, 0.5, 0.5), (25, 29, 0.5, 0.5)]
# Remote-controlled sectionalizing switches (declared assumption of the case study)
_IEEE33_SECTIONALIZERS = {(1, 2), (2, 3), (3, 23), (5, 6), (6, 26), (8, 9), (12, 13),
                          (2, 19), (15, 16), (28, 29), (31, 32), (24, 25)}


def ieee33(H=12, dt_h=5 / 60, seed=7):
    """IEEE 33-bus feeder with an assumed DER/BESS/EV overlay (declared in the paper)."""
    S_base, V_base = 1000.0, 12.66
    Zb = V_base ** 2 / (S_base / 1000.0)  # ohm
    edges, loads = [], {}
    for k, (i, j, R, X, P, Q) in enumerate(_IEEE33):
        trunk = j <= 6 or (i, j) in {(2, 19), (3, 23)}
        edges.append(Edge(k, i, j, R / Zb, X / Zb, 4000.0 if trunk else 2000.0,
                          switchable=(i, j) in _IEEE33_SECTIONALIZERS))
        loads[j] = Load(P, Q)
    for m, (i, j, R, X) in enumerate(_IEEE33_TIES):
        edges.append(Edge(len(_IEEE33) + m, i, j, R / Zb, X / Zb, 2000.0,
                          switchable=True, normally_open=True))
    tier1 = {7, 14, 31}           # hospital, water pumping, emergency services (assumed)
    tier2 = {8, 24, 25, 30, 32}
    for b, L in loads.items():
        L.tier = 1 if b in tier1 else (2 if b in tier2 else 3)
        L.rho = 0.0 if L.tier == 1 else (0.2 if L.tier == 2 else 0.3)
    T = np.arange(1, H + 1)
    pv = np.clip(0.75 - 0.03 * (T - 1), 0.3, 1.0)        # declining afternoon PV forecast
    wind = 0.55 + 0.05 * np.sin(T / 3.0)
    res = [
        Resource("G1-diesel", 7, 'D', Pmax=450, Pmin=90, Qmax=300, rated=450, droop=0.04,
                 BS=1, GFM=1, FR=1, VR=1, PROT=1, If=5 * 0.45, cdeg=0.0),
        Resource("G2-microturbine", 30, 'D', Pmax=350, Pmin=70, Qmax=250, rated=350, droop=0.04,
                 BS=1, GFM=1, FR=1, VR=1, PROT=1, If=4 * 0.35, cdeg=0.0),
        Resource("PV1", 14, 'R', Pmax=300, rated=300, Qmax=100, GFL=1, VR=1, If=1.2 * 0.3,
                 profile=pv, sigma=0.15),
        Resource("PV2", 25, 'R', Pmax=350, rated=350, Qmax=120, GFL=1, VR=1, If=1.2 * 0.35,
                 profile=pv, sigma=0.15),
        Resource("PV3", 32, 'R', Pmax=250, rated=250, Qmax=80, GFL=1, VR=1, If=1.2 * 0.25,
                 profile=pv, sigma=0.15),
        Resource("WT1", 22, 'R', Pmax=200, rated=200, Qmax=60, GFL=1, VR=1, If=1.2 * 0.2,
                 profile=wind, sigma=0.20),
        Resource("BESS1", 18, 'B', Pmax=250, Pch=250, rated=250, Qmax=150, droop=0.01, E=750,
                 BS=1, GFM=1, FR=1, VR=1, PROT=1, If=1.5 * 0.25, soc0=0.85),
        Resource("BESS2", 24, 'B', Pmax=300, Pch=300, rated=300, Qmax=180, droop=0.01, E=900,
                 BS=1, GFM=1, FR=1, VR=1, PROT=1, If=1.5 * 0.30, soc0=0.80),
    ]
    rng = np.random.default_rng(seed)
    for e in range(10):              # EV fleet at a depot on bus 29 (V2G, grid-following)
        tdep = float(rng.choice([0.5, 0.75, 1.0, 2.0, 3.0]))
        res.append(Resource(f"EV{e+1}", 29, 'V', Pmax=11, Pch=11, rated=11, E=60, GFL=1, FR=1,
                            droop=0.05, soc0=float(rng.uniform(0.6, 0.9)), Tdep=tdep,
                            socreq=0.6, pi=0.85, avail=np.ones(H), socmin=0.2))
    return Network("IEEE33-DER", list(range(1, 34)), edges, loads, res, S_base, V_base)


def small_test_net(kind="two_bus"):
    """Tiny networks used by the verification suite."""
    S_base = 1000.0
    if kind == "two_bus":
        edges = [Edge(0, 1, 2, 0.02, 0.03, 1500.0, switchable=True)]
        loads = {2: Load(150, 50, tier=1)}
        res = [Resource("G", 1, 'D', Pmax=1000, Pmin=0, Qmax=600, rated=1000, BS=1, GFM=1, FR=1,
                        VR=1, PROT=1, If=5.0, droop=0.05)]
        return Network("two-bus", [1, 2], edges, loads, res, S_base)
    raise ValueError(kind)
