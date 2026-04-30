from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .active_set import ActiveSet, active_kkt_equalities
from .kkt import build_active_kkt_system
from .solution import OPFSolution


@dataclass
class RegionQuadraticPath:
    """Per-sample fixed-active-set paths for one RHS-block displacement.

    For alpha, rbias, and gamma, the OPF matrix is fixed.  Conditional on the
    active set of sample q being unchanged, its solution has the affine form
        x_q(tau) = x_q(0) + tau dx_q,
    and its dispatch loss is exactly quadratic in tau.  The field tau_samples[q]
    is the certified no-switch radius for that sample only.  Step 10 uses this
    per-sample radius: safe samples are evaluated by the affine path; only unsafe
    samples are re-solved by a full forward QP.
    """
    tau_region: float
    tau_samples: np.ndarray
    c0_samples: np.ndarray
    c1_samples: np.ndarray
    c2_samples: np.ndarray
    x0: list[np.ndarray]
    dx: list[np.ndarray]
    sample_ids: np.ndarray
    status_counts: dict

    def sample_loss(self, tau: float) -> np.ndarray:
        t = float(tau)
        return self.c0_samples + self.c1_samples * t + self.c2_samples * t * t

    def loss(self, tau: float, mask: np.ndarray | None = None) -> float:
        vals = self.sample_loss(tau)
        return float(np.mean(vals if mask is None else vals[np.asarray(mask, dtype=bool)]))

    def sol(self, template, loc: int, tau: float) -> OPFSolution:
        xt = np.asarray(self.x0[loc] + float(tau) * self.dx[loc], dtype=float)
        pg, theta, sp, sm = template.split(xt)
        obj = float(0.5 * xt @ (template.P @ xt) + template.q @ xt)
        return OPFSolution(x=xt.copy(), pg=pg.copy(), theta=theta.copy(),
                           sp=sp.copy(), sm=sm.copy(), obj=obj,
                           status="fixed_active_quad", iter=0)

    def sols(self, template, tau: float) -> list[OPFSolution]:
        return [self.sol(template, k, tau) for k in range(len(self.sample_ids))]


def _active_rhs_delta(template, active: ActiveSet, block: str,
                      step: np.ndarray, loss_hat: float) -> np.ndarray:
    """RHS perturbation of G_A x = h_A for one full block step."""
    n, m = template.net.n, template.net.m
    dh = np.zeros(n + active.counts()["total_ineq"])
    step = np.asarray(step, dtype=float).reshape(-1)

    if block == "alpha":
        dh[:n] = float(loss_hat) * step[:n]
        return dh
    if block == "rbias":
        dh[:n] = step[:n]
        return dh
    if block == "gamma":
        gp, gm = step[:m], step[m:]
        k = n
        k += int(active.pg_min.sum())
        k += int(active.pg_max.sum())
        ids = np.flatnonzero(active.line_p)
        dh[k:k + len(ids)] = -float(loss_hat) * gp[ids]
        k += len(ids)
        ids = np.flatnonzero(active.line_m)
        dh[k:k + len(ids)] = -float(loss_hat) * gm[ids]
        return dh
    raise ValueError(block)


def _full_bound_delta(template, block: str, step: np.ndarray, loss_hat: float):
    """Perturbations dl,du of all bounds l <= A x <= u."""
    rows, n, m = template.rows, template.net.n, template.net.m
    dl = np.zeros(template.nc)
    du = np.zeros(template.nc)
    step = np.asarray(step, dtype=float).reshape(-1)
    if block == "alpha":
        du[rows.bal] = float(loss_hat) * step[:n]
        dl[rows.bal] = du[rows.bal]
    elif block == "rbias":
        du[rows.bal] = step[:n]
        dl[rows.bal] = du[rows.bal]
    elif block == "gamma":
        gp, gm = step[:m], step[m:]
        du[rows.line_p] = -float(loss_hat) * gp
        du[rows.line_m] = -float(loss_hat) * gm
    else:
        raise ValueError(block)
    return dl, du


def _tau_from_distance(dist: np.ndarray, ddist: np.ndarray, margin: float) -> float:
    mask = ddist < -1e-14
    if not np.any(mask):
        return np.inf
    vals = (dist[mask] - margin) / (-ddist[mask])
    return max(0.0, float(vals.min()))


def _primal_tau(problem, sid: int, active: ActiveSet, x0: np.ndarray, dx: np.ndarray,
                block: str, step: np.ndarray, margin: float) -> float:
    tpl, rows = problem.template, problem.template.rows
    A = tpl.build_A(problem.params.b)
    l, u = tpl.bounds(problem.ds.pd[int(sid)], problem.params, float(problem.loss_hat[int(sid)]))
    dl, du = _full_bound_delta(tpl, block, step, float(problem.loss_hat[int(sid)]))
    ax, dax = A @ x0, A @ dx
    tau = np.inf

    def lower(row_slice, inactive):
        ids = np.arange(row_slice.start, row_slice.stop)[inactive]
        return np.inf if ids.size == 0 else _tau_from_distance(ax[ids] - l[ids], dax[ids] - dl[ids], margin)

    def upper(row_slice, inactive):
        ids = np.arange(row_slice.start, row_slice.stop)[inactive]
        return np.inf if ids.size == 0 else _tau_from_distance(u[ids] - ax[ids], du[ids] - dax[ids], margin)

    tau = min(tau, lower(rows.pg, ~active.pg_min))
    tau = min(tau, upper(rows.pg, ~active.pg_max))
    tau = min(tau, upper(rows.line_p, ~active.line_p))
    tau = min(tau, upper(rows.line_m, ~active.line_m))
    tau = min(tau, lower(rows.sp, ~active.sp_zero))
    tau = min(tau, lower(rows.sm, ~active.sm_zero))
    return tau


def _dual_tau(mu: np.ndarray, dmu: np.ndarray, labels: list[str], dual_tol: float) -> float:
    tau = np.inf
    for k, lab in enumerate(labels):
        if lab == "balance":
            continue
        if lab in {"pg_min", "sp_zero", "sm_zero"}:      # lower-bound multiplier <= 0
            if dmu[k] > 1e-14:
                tau = min(tau, max(0.0, float((-dual_tol - mu[k]) / dmu[k])))
        else:                                             # upper-bound multiplier >= 0
            if dmu[k] < -1e-14:
                tau = min(tau, max(0.0, float((dual_tol - mu[k]) / dmu[k])))
    return tau


def build_region_quadratic_path(problem, sample_ids, active_sets: list[ActiveSet],
                                block: str, step: np.ndarray,
                                *, margin: float = 1e-8,
                                dual_tol: float = 1e-8) -> RegionQuadraticPath:
    """Build per-sample fixed-active-set affine paths for one RHS block."""
    if block not in {"alpha", "rbias", "gamma"}:
        raise ValueError("Region quadratic line search is only for alpha, rbias, gamma.")

    ids = np.asarray(sample_ids, dtype=int)
    groups = defaultdict(list)
    for loc, act in enumerate(active_sets):
        groups[act.key()].append(loc)

    n = len(ids)
    x0_list = [None] * n
    dx_list = [None] * n
    tau_samples = np.full(n, np.inf)
    c0 = np.zeros(n)
    c1 = np.zeros(n)
    c2 = np.zeros(n)

    for locs in groups.values():
        rep = locs[0]
        sid_rep = int(ids[rep])
        act_rep = active_sets[rep]
        system = build_active_kkt_system(problem.template, act_rep,
                                         problem.ds.pd[sid_rep], problem.params,
                                         float(problem.loss_hat[sid_rep]))
        for loc in locs:
            sid = int(ids[loc])
            active = active_sets[loc]
            _, h, labels = active_kkt_equalities(problem.template, active,
                                                 problem.ds.pd[sid], problem.params,
                                                 float(problem.loss_hat[sid]))
            x0, mu0 = system.solve_kkt(h=h)
            dh = _active_rhs_delta(problem.template, active, block, step,
                                   float(problem.loss_hat[sid]))
            dx, dmu = system.sensitivity(dh)

            tau_samples[loc] = min(_primal_tau(problem, sid, active, x0, dx, block, step, margin),
                                   _dual_tau(mu0, dmu, labels, dual_tol))

            err0 = x0[problem.template.v.pg] - problem.ds.pg_ac[sid]
            dpg = dx[problem.template.v.pg]
            c0[loc] = 0.5 * float(err0 @ err0)
            c1[loc] = float(err0 @ dpg)
            c2[loc] = 0.5 * float(dpg @ dpg)
            x0_list[loc] = x0
            dx_list[loc] = dx

    return RegionQuadraticPath(tau_region=float(np.min(tau_samples)),
                               tau_samples=tau_samples,
                               c0_samples=c0,
                               c1_samples=c1,
                               c2_samples=c2,
                               x0=x0_list,
                               dx=dx_list,
                               sample_ids=ids,
                               status_counts={"fixed_active_quad": n})
