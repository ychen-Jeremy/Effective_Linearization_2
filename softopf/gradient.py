from dataclasses import dataclass
import numpy as np
from .active_set import ActiveSet, group_active_sets, active_kkt_equalities
from .kkt import ActiveKKTSystem, build_active_kkt_system
from .solution import OPFSolution


@dataclass
class BatchGradient:
    loss: float
    grad_alpha: np.ndarray
    grad_rbias: np.ndarray
    grad_b: np.ndarray
    grad_gamma_p: np.ndarray
    grad_gamma_m: np.ndarray
    group_sizes: list[int]
    n_groups: int


def dispatch_loss_grad_x(template, sol: OPFSolution, pg_ac: np.ndarray) -> tuple[float, np.ndarray]:
    """Return 0.5 ||Pg - PgAC||_2^2 and its gradient with respect to x."""
    diff = sol.pg - np.asarray(pg_ac, dtype=float).reshape(-1)
    grad_x = np.zeros(template.nx)
    grad_x[template.v.pg] = diff
    return 0.5 * float(diff @ diff), grad_x


def rhs_grad_from_adjoint(system: ActiveKKTSystem, grad_x: np.ndarray) -> np.ndarray:
    """Gradient of the loss with respect to the balance RHS."""
    _, w_h = system.adjoint(grad_x)
    return w_h[:system.template.net.n]


def _b_gamma_grads(system: ActiveKKTSystem, active: ActiveSet, x: np.ndarray,
                   mu: np.ndarray, w_x: np.ndarray, w_h: np.ndarray,
                   loss_hat: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adjoint gradients for b, gamma_p, and gamma_m under a fixed active set.

    If K z = rhs and K depends on b, then
      d ell = - w_x' (dG') mu - w_h' (dG x) + w_h' dh.
    gamma enters only h for active line-limit rows; b enters the balance and
    active line-flow rows.
    """
    tpl, net = system.template, system.template.net
    theta = x[tpl.v.theta]
    wx_theta = w_x[tpl.v.theta]
    y = net.Ar @ theta
    wy = net.Ar @ wx_theta

    grad_b = np.zeros(net.m)
    grad_gp = np.zeros(net.m)
    grad_gm = np.zeros(net.m)

    # Balance rows: d(Cg Pg - A' diag(b) A_r theta)/db_l = -A_l' y_l.
    mu_bal = mu[:net.n]
    wh_bal = w_h[:net.n]
    Amu = net.A @ mu_bal
    Awh = net.A @ wh_bal
    grad_b += Amu * wy + Awh * y

    labels = np.asarray(system.labels)

    # Active positive line rows: dGx/db_l = y_l, dh/dgamma_p_l = -loss_hat.
    rows_p = np.flatnonzero(labels == "line_p")
    ids_p = np.flatnonzero(active.line_p)
    for row, ell in zip(rows_p, ids_p):
        grad_b[ell] += -mu[row] * wy[ell] - w_h[row] * y[ell]
        grad_gp[ell] += -float(loss_hat) * w_h[row]

    # Active negative line rows: dGx/db_l = -y_l, dh/dgamma_m_l = -loss_hat.
    rows_m = np.flatnonzero(labels == "line_m")
    ids_m = np.flatnonzero(active.line_m)
    for row, ell in zip(rows_m, ids_m):
        grad_b[ell] += mu[row] * wy[ell] + w_h[row] * y[ell]
        grad_gm[ell] += -float(loss_hat) * w_h[row]

    return grad_b, grad_gp, grad_gm


def sample_loss_and_grads(problem, sample_id: int, sol: OPFSolution,
                          active: ActiveSet) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    system = build_active_kkt_system(problem.template, active, problem.ds.pd[sample_id],
                                     problem.params, problem.loss_hat[sample_id])
    _, h, _ = active_kkt_equalities(problem.template, active, problem.ds.pd[sample_id],
                                    problem.params, problem.loss_hat[sample_id])
    x_kkt, mu = system.solve_kkt(h)
    loss, grad_x = dispatch_loss_grad_x(problem.template, sol, problem.ds.pg_ac[sample_id])
    w_x, w_h = system.adjoint(grad_x)
    g_rhs = w_h[:problem.net.n]
    gb, ggp, ggm = _b_gamma_grads(system, active, x_kkt, mu, w_x, w_h, problem.loss_hat[sample_id])
    return loss, float(problem.loss_hat[sample_id]) * g_rhs, g_rhs, gb, ggp, ggm


def batch_loss_and_grads(problem, sample_ids, sols: list[OPFSolution],
                         active_sets: list[ActiveSet]) -> BatchGradient:
    """Mean dispatch loss and all Step-7 block gradients.

    Samples with the same active-set signature share one sparse KKT
    factorization.  The RHS and multipliers are recomputed per sample.
    """
    sample_ids = np.asarray(sample_ids, dtype=int)
    groups = group_active_sets(active_sets)
    nbus, m = problem.net.n, problem.net.m
    grad_alpha = np.zeros(nbus)
    grad_rbias = np.zeros(nbus)
    grad_b = np.zeros(m)
    grad_gp = np.zeros(m)
    grad_gm = np.zeros(m)
    loss_sum = 0.0

    for local_ids in groups.values():
        rep = local_ids[0]
        sid_rep = int(sample_ids[rep])
        active = active_sets[rep]
        system = build_active_kkt_system(problem.template, active, problem.ds.pd[sid_rep],
                                         problem.params, problem.loss_hat[sid_rep])
        for loc in local_ids:
            sid = int(sample_ids[loc])
            _, h, _ = active_kkt_equalities(problem.template, active, problem.ds.pd[sid],
                                            problem.params, problem.loss_hat[sid])
            x_kkt, mu = system.solve_kkt(h)
            loss, grad_x = dispatch_loss_grad_x(problem.template, sols[loc], problem.ds.pg_ac[sid])
            w_x, w_h = system.adjoint(grad_x)
            g_rhs = w_h[:nbus]
            gb, ggp, ggm = _b_gamma_grads(system, active, x_kkt, mu, w_x, w_h, problem.loss_hat[sid])

            loss_sum += loss
            grad_alpha += float(problem.loss_hat[sid]) * g_rhs
            grad_rbias += g_rhs
            grad_b += gb
            grad_gp += ggp
            grad_gm += ggm

    n = len(sample_ids)
    return BatchGradient(loss=loss_sum / n,
                         grad_alpha=grad_alpha / n,
                         grad_rbias=grad_rbias / n,
                         grad_b=grad_b / n,
                         grad_gamma_p=grad_gp / n,
                         grad_gamma_m=grad_gm / n,
                         group_sizes=sorted((len(v) for v in groups.values()), reverse=True),
                         n_groups=len(groups))


# Backward-compatible helpers used by earlier scripts.
def batch_loss_and_alpha_grad(problem, sample_ids, sols, active_sets):
    return batch_loss_and_grads(problem, sample_ids, sols, active_sets)


def alpha_gradient_from_adjoint(system: ActiveKKTSystem, grad_x: np.ndarray, loss_hat: float) -> np.ndarray:
    return float(loss_hat) * rhs_grad_from_adjoint(system, grad_x)


def rbias_gradient_from_adjoint(system: ActiveKKTSystem, grad_x: np.ndarray) -> np.ndarray:
    return rhs_grad_from_adjoint(system, grad_x)
