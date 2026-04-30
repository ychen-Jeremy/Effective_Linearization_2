from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .active_set import ActiveSet, active_kkt_equalities
from .kkt import build_active_kkt_system
from .solution import OPFSolution


@dataclass
class FixedActiveEndpoint:
    """Endpoint information along one proposed parameter step.

    The candidate is valid without a forward QP solve only if ``safe`` is true.
    In that case, all samples stay in the same critical region for the whole
    segment from the current parameter to the candidate.
    """
    safe: bool
    tau_region: float
    loss: float
    sols: list[OPFSolution]
    status_counts: dict


def _solution_from_x(template, x: np.ndarray, obj: float) -> OPFSolution:
    pg, theta, sp, sm = template.split(np.asarray(x, dtype=float))
    return OPFSolution(x=np.asarray(x, dtype=float).copy(), pg=pg.copy(), theta=theta.copy(),
                       sp=sp.copy(), sm=sm.copy(), obj=float(obj),
                       status="fixed_active_kkt", iter=0)


def _qp_obj(template, x: np.ndarray) -> float:
    return float(0.5 * x @ (template.P @ x) + template.q @ x)


def active_rhs_delta(template, active: ActiveSet, block: str,
                     step: np.ndarray, loss_hat: float) -> np.ndarray:
    """RHS perturbation of the active equality system G_A x = h_A."""
    n, m = template.net.n, template.net.m
    dh = np.zeros(n + active.counts()["total_ineq"])
    step = np.asarray(step, dtype=float).reshape(-1)

    if block == "alpha":
        dh[:n] = float(loss_hat) * step
        return dh
    if block == "rbias":
        dh[:n] = step
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
    raise ValueError(f"critical-region check is implemented only for RHS blocks, got {block}")


def full_bound_delta(template, block: str, step: np.ndarray, loss_hat: float):
    """Perturbations dl, du of the full bound vector l <= A x <= u."""
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
        raise ValueError(f"critical-region check is implemented only for RHS blocks, got {block}")
    return dl, du


def _tau_from_distance(dist: np.ndarray, ddist: np.ndarray, margin: float) -> float:
    tau = np.inf
    mask = ddist < -1e-14
    if np.any(mask):
        vals = (dist[mask] - margin) / (-ddist[mask])
        if vals.size:
            tau = min(tau, max(0.0, float(vals.min())))
    return tau


def sample_primal_region_bound(problem, sid: int, active: ActiveSet,
                               x0: np.ndarray, dx: np.ndarray, block: str,
                               step: np.ndarray, margin: float) -> float:
    tpl, rows = problem.template, problem.template.rows
    A = tpl.build_A(problem.params.b)
    l, u = tpl.bounds(problem.ds.pd[int(sid)], problem.params, float(problem.loss_hat[int(sid)]))
    dl, du = full_bound_delta(tpl, block, step, float(problem.loss_hat[int(sid)]))
    ax = A @ x0
    dax = A @ dx
    tau = np.inf

    def lower(bound_slice, inactive_mask):
        ids = np.arange(bound_slice.start, bound_slice.stop)[inactive_mask]
        if ids.size == 0:
            return np.inf
        dist = ax[ids] - l[ids]
        ddist = dax[ids] - dl[ids]
        return _tau_from_distance(dist, ddist, margin)

    def upper(bound_slice, inactive_mask):
        ids = np.arange(bound_slice.start, bound_slice.stop)[inactive_mask]
        if ids.size == 0:
            return np.inf
        dist = u[ids] - ax[ids]
        ddist = du[ids] - dax[ids]
        return _tau_from_distance(dist, ddist, margin)

    tau = min(tau, lower(rows.pg, ~active.pg_min))
    tau = min(tau, upper(rows.pg, ~active.pg_max))
    tau = min(tau, upper(rows.line_p, ~active.line_p))
    tau = min(tau, upper(rows.line_m, ~active.line_m))
    tau = min(tau, lower(rows.sp, ~active.sp_zero))
    tau = min(tau, lower(rows.sm, ~active.sm_zero))
    return tau


def dual_region_bound(mu: np.ndarray, dmu: np.ndarray, labels: list[str], dual_tol: float) -> float:
    """Largest lambda before an active inequality multiplier changes sign."""
    tau = np.inf
    for k, lab in enumerate(labels):
        if lab == "balance":
            continue
        if lab in {"pg_min", "sp_zero", "sm_zero"}:      # lower-bound multiplier should be nonpositive
            if dmu[k] > 1e-14:
                val = (-dual_tol - mu[k]) / dmu[k]
                tau = min(tau, max(0.0, float(val)))
        else:                                             # upper-bound multiplier should be nonnegative
            if dmu[k] < -1e-14:
                val = (dual_tol - mu[k]) / dmu[k]
                tau = min(tau, max(0.0, float(val)))
    return tau


def fixed_active_endpoint(problem, sample_ids, sols, active_sets, block: str,
                          step: np.ndarray, *, margin: float = 1e-8,
                          dual_tol: float = 1e-8, safety: float = 0.995) -> FixedActiveEndpoint:
    """Evaluate a proposed RHS-block endpoint without solving forward QPs.

    Returns ``safe=True`` only if the whole step lies strictly inside the
    current critical region for every sample.  Otherwise the caller should fall
    back to a true forward QP solve for that trial point.
    """
    if block not in {"alpha", "rbias", "gamma"}:
        raise ValueError("fixed-active endpoint is only available for alpha, rbias, gamma")

    ids = np.asarray(sample_ids, dtype=int)
    groups = defaultdict(list)
    for loc, act in enumerate(active_sets):
        groups[act.key()].append(loc)

    pred_sols = [None] * len(ids)
    losses = []
    tau_region = np.inf

    for locs in groups.values():
        rep_loc = locs[0]
        rep_sid = int(ids[rep_loc])
        rep_active = active_sets[rep_loc]
        system = build_active_kkt_system(problem.template, rep_active,
                                         problem.ds.pd[rep_sid], problem.params,
                                         float(problem.loss_hat[rep_sid]))
        for loc in locs:
            sid = int(ids[loc])
            act = active_sets[loc]
            _, h, labels = active_kkt_equalities(problem.template, act,
                                                 problem.ds.pd[sid], problem.params,
                                                 float(problem.loss_hat[sid]))
            x0, mu0 = system.solve_kkt(h=h)
            dh = active_rhs_delta(problem.template, act, block, step,
                                  float(problem.loss_hat[sid]))
            dx, dmu = system.sensitivity(dh)
            tau_p = sample_primal_region_bound(problem, sid, act, x0, dx, block, step, margin)
            tau_d = dual_region_bound(mu0, dmu, labels, dual_tol)
            tau_region = min(tau_region, tau_p, tau_d)
            x1 = x0 + dx
            losses.append(dispatch_loss_from_x(problem, sid, x1))
            pred_sols[loc] = _solution_from_x(problem.template, x1, _qp_obj(problem.template, x1))

    safe = bool(1.0 <= safety * tau_region)
    return FixedActiveEndpoint(safe=safe, tau_region=float(tau_region),
                               loss=float(np.mean(losses)), sols=pred_sols,
                               status_counts={"fixed_active_kkt": len(ids)})


def dispatch_loss_from_x(problem, sid: int, x: np.ndarray) -> float:
    pg = x[problem.template.v.pg]
    err = pg - problem.ds.pg_ac[int(sid)]
    return float(0.5 * np.dot(err, err))
