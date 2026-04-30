from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

# 0-based MATPOWER column constants.
BUS_I = 0
GEN_BUS, PMAX, PMIN = 0, 8, 9
F_BUS, T_BUS, BR_X, RATE_A, TAP = 0, 1, 3, 5, 8
COST = 4


@dataclass
class Network:
    base_mva: float
    n: int
    m: int
    g: int
    A: sp.csr_matrix       # m x n, +1 at from bus, -1 at to bus
    Ar: sp.csr_matrix      # m x (n-1), reference column removed
    Cg: sp.csr_matrix      # n x g
    Bf: sp.csr_matrix      # m x (n-1), MW/rad
    Bbus: sp.csr_matrix    # n x (n-1), MW/rad
    bphys: np.ndarray
    fmax: np.ndarray
    pg_min: np.ndarray
    pg_max: np.ndarray
    c2: np.ndarray
    c1: np.ndarray
    ref: int
    nonref: np.ndarray
    active_gen_idx: np.ndarray
    active_branch_idx: np.ndarray


def _idx1(x) -> np.ndarray:
    return np.asarray(x, dtype=int).reshape(-1) - 1


def build_network(mpc0: dict, meta: dict) -> Network:
    bus = np.asarray(mpc0["bus"], dtype=float)
    gen_all = np.asarray(mpc0["gen"], dtype=float)
    branch_all = np.asarray(mpc0["branch"], dtype=float)
    gencost_all = np.asarray(mpc0["gencost"], dtype=float)
    base_mva = float(np.asarray(mpc0["baseMVA"]).squeeze())

    active_gen_idx = _idx1(meta["active_gen_idx"])
    active_branch_idx = _idx1(meta["active_branch_idx"])
    ref = int(_idx1(meta["ref_bus_idx"])[0])

    gen = gen_all[active_gen_idx]
    branch = branch_all[active_branch_idx]
    gencost = gencost_all[active_gen_idx]

    n, m, g = bus.shape[0], branch.shape[0], gen.shape[0]
    bus_id_to_row = {int(bus[i, BUS_I]): i for i in range(n)}

    rows = np.arange(m)
    f = np.array([bus_id_to_row[int(v)] for v in branch[:, F_BUS]])
    t = np.array([bus_id_to_row[int(v)] for v in branch[:, T_BUS]])
    A = sp.csr_matrix((np.r_[np.ones(m), -np.ones(m)],
                       (np.r_[rows, rows], np.r_[f, t])), shape=(m, n))

    nonref = np.array([i for i in range(n) if i != ref], dtype=int)
    Ar = A[:, nonref].tocsr()

    tap = branch[:, TAP].copy()
    tap[tap == 0.0] = 1.0
    bphys = base_mva / (branch[:, BR_X] * tap)
    Bf = sp.diags(bphys, format="csr") @ Ar
    Bbus = A.T @ Bf

    gen_bus = np.array([bus_id_to_row[int(v)] for v in gen[:, GEN_BUS]])
    Cg = sp.csr_matrix((np.ones(g), (gen_bus, np.arange(g))), shape=(n, g))

    return Network(
        base_mva=base_mva, n=n, m=m, g=g, A=A, Ar=Ar,
        Cg=Cg.tocsr(), Bf=Bf.tocsr(), Bbus=Bbus.tocsr(), bphys=bphys,
        fmax=branch[:, RATE_A], pg_min=gen[:, PMIN], pg_max=gen[:, PMAX],
        c2=gencost[:, COST], c1=gencost[:, COST + 1], ref=ref, nonref=nonref,
        active_gen_idx=active_gen_idx, active_branch_idx=active_branch_idx,
    )
