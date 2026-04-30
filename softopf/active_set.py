from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
from .network import Network
from .params import Params
from .solution import OPFSolution
from .template import SoftQPTemplate


@dataclass
class ActiveSet:
    """Primal active-set signature for the softened OPF QP."""
    pg_min: np.ndarray
    pg_max: np.ndarray
    line_p: np.ndarray
    line_m: np.ndarray
    sp_zero: np.ndarray
    sm_zero: np.ndarray

    def key(self) -> tuple[bytes, ...]:
        return tuple(np.packbits(v.astype(np.uint8)).tobytes()
                     for v in (self.pg_min, self.pg_max, self.line_p,
                               self.line_m, self.sp_zero, self.sm_zero))

    def counts(self) -> dict:
        return {
            "pg_min": int(self.pg_min.sum()),
            "pg_max": int(self.pg_max.sum()),
            "line_p": int(self.line_p.sum()),
            "line_m": int(self.line_m.sum()),
            "sp_zero": int(self.sp_zero.sum()),
            "sm_zero": int(self.sm_zero.sum()),
            "total_ineq": int(self.pg_min.sum() + self.pg_max.sum()
                              + self.line_p.sum() + self.line_m.sum()
                              + self.sp_zero.sum() + self.sm_zero.sum()),
        }


def extract_active_set(net: Network, sol: OPFSolution, tol: float = 1e-5,
                       params: Params | None = None, loss_hat: float = 0.0) -> ActiveSet:
    b = net.bphys if params is None else params.b
    gp = np.zeros(net.m) if params is None else params.gamma_p
    gm = np.zeros(net.m) if params is None else params.gamma_m
    flow = b * (net.Ar @ sol.theta)
    fmax_p = net.fmax - float(loss_hat) * gp
    fmax_m = net.fmax - float(loss_hat) * gm
    return ActiveSet(
        pg_min=sol.pg <= net.pg_min + tol,
        pg_max=sol.pg >= net.pg_max - tol,
        line_p=flow - sol.sp >= fmax_p - tol,
        line_m=-flow - sol.sm >= fmax_m - tol,
        sp_zero=sol.sp <= tol,
        sm_zero=sol.sm <= tol,
    )


def group_active_sets(active_sets: list[ActiveSet]) -> dict[tuple[bytes, ...], list[int]]:
    groups = defaultdict(list)
    for i, a in enumerate(active_sets):
        groups[a.key()].append(i)
    return dict(groups)


def group_size_counts(active_sets: list[ActiveSet]) -> Counter:
    return Counter(a.key() for a in active_sets)


def representative_counts(active_sets: list[ActiveSet]) -> list[dict]:
    groups = group_active_sets(active_sets)
    reps = []
    for key, idx in groups.items():
        d = active_sets[idx[0]].counts()
        d["size"] = len(idx)
        reps.append(d)
    return sorted(reps, key=lambda x: -x["size"])


def active_kkt_equalities(template: SoftQPTemplate, active: ActiveSet,
                          pd: np.ndarray, params: Params, loss_hat: float):
    """Return G_A, h_A, labels for fixed-active-set reconstruction."""
    r = template.rows
    A_full = template.build_A(params.b)
    l, u = template.bounds(pd, params, loss_hat)
    row_idx = list(range(r.bal.start, r.bal.stop))
    rhs = list(u[r.bal])
    labels = ["balance"] * template.net.n

    def add(mask, base, side, name):
        ids = base + np.flatnonzero(mask)
        row_idx.extend(ids.tolist())
        rhs.extend((l[ids] if side == "lower" else u[ids]).tolist())
        labels.extend([name] * len(ids))

    add(active.pg_min, r.pg.start, "lower", "pg_min")
    add(active.pg_max, r.pg.start, "upper", "pg_max")
    add(active.line_p, r.line_p.start, "upper", "line_p")
    add(active.line_m, r.line_m.start, "upper", "line_m")
    add(active.sp_zero, r.sp.start, "lower", "sp_zero")
    add(active.sm_zero, r.sm.start, "lower", "sm_zero")

    return A_full[row_idx, :].tocsr(), np.asarray(rhs, dtype=float), labels


def active_residuals(net: Network, sol: OPFSolution, active: ActiveSet,
                     params: Params | None = None, loss_hat: float = 0.0) -> dict:
    """Distances to the selected active constraints."""
    b = net.bphys if params is None else params.b
    gp = np.zeros(net.m) if params is None else params.gamma_p
    gm = np.zeros(net.m) if params is None else params.gamma_m
    flow = b * (net.Ar @ sol.theta)
    fmax_p = net.fmax - float(loss_hat) * gp
    fmax_m = net.fmax - float(loss_hat) * gm

    def max_abs(v):
        return 0.0 if v.size == 0 else float(np.max(np.abs(v)))

    return {
        "pg_min": max_abs(sol.pg[active.pg_min] - net.pg_min[active.pg_min]),
        "pg_max": max_abs(sol.pg[active.pg_max] - net.pg_max[active.pg_max]),
        "line_p": max_abs((flow - sol.sp - fmax_p)[active.line_p]),
        "line_m": max_abs((-flow - sol.sm - fmax_m)[active.line_m]),
        "sp_zero": max_abs(sol.sp[active.sp_zero]),
        "sm_zero": max_abs(sol.sm[active.sm_zero]),
    }
