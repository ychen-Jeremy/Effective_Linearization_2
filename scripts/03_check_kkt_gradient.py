from pathlib import Path
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from softopf.config import Settings
from softopf.pipeline import build_problem, make_solver, solve_one
from softopf.active_set import extract_active_set, active_residuals
from softopf.kkt import build_active_kkt_system, active_rhs_delta_for_alpha, kkt_residual_norm
from softopf.gradient import dispatch_loss_grad_x, alpha_gradient_from_adjoint


def resolve(path):
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def feasible_direction(alpha):
    pos = np.flatnonzero(alpha > 1e-8)
    i = pos[np.argmax(alpha[pos])]
    j = pos[np.argsort(alpha[pos])[-2]] if len(pos) > 1 else i
    d = np.zeros_like(alpha)
    d[i], d[j] = 1.0, -1.0
    return d, int(i), int(j)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default="data/case39.mat")
    p.add_argument("--case", default=None)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--backend", default="cvxpy", choices=["cvxpy", "osqp"])
    p.add_argument("--solver", default="auto")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--eps", type=float, default=1e-4)
    args = p.parse_args()

    st = Settings(cvxpy_solver=args.solver, active_tol=args.tol)
    prob = build_problem(resolve(args.mat), resolve(args.case), st)
    solver = make_solver(prob, args.backend)

    sid = args.sample
    sol = solve_one(prob, solver, sid, args.backend)
    act = extract_active_set(prob.net, sol, tol=st.active_tol, params=prob.params, loss_hat=float(prob.loss_hat[sid]))
    sys_kkt = build_active_kkt_system(prob.template, act, prob.ds.pd[sid],
                                      prob.params, prob.loss_hat[sid])

    x_hat, mu_hat = sys_kkt.solve_kkt()
    kkt_res = kkt_residual_norm(sys_kkt, x_hat, mu_hat)
    active_res = active_residuals(prob.net, sol, act, params=prob.params, loss_hat=float(prob.loss_hat[sid]))

    print("[Reduced-KKT reconstruction]")
    print(f"  sample / backend        : {sid} / {args.backend}")
    print(f"  forward status          : {sol.status}")
    print(f"  KKT matrix shape / nnz  : {sys_kkt.K.shape} / {sys_kkt.K.nnz}")
    print(f"  active rows             : {sys_kkt.na}")
    print(f"  ||x_kkt - x_solve||_inf : {np.linalg.norm(x_hat - sol.x, ord=np.inf):.3e}")
    print(f"  stationarity_inf        : {kkt_res['stationarity_inf']:.3e}")
    print(f"  active_primal_inf       : {kkt_res['active_primal_inf']:.3e}")
    print(f"  active residual max     : {max(active_res.values()):.3e}")

    alpha0 = prob.params.alpha.copy()
    d_alpha, i, j = feasible_direction(alpha0)
    eps = min(args.eps, 0.25 * alpha0[j])
    loss0, grad_x = dispatch_loss_grad_x(prob.template, sol, prob.ds.pg_ac[sid])
    grad_alpha = alpha_gradient_from_adjoint(sys_kkt, grad_x, prob.loss_hat[sid])
    dh = active_rhs_delta_for_alpha(prob.template, eps * d_alpha, prob.loss_hat[sid], sys_kkt.na)
    dx, _ = sys_kkt.sensitivity(dh)
    pred_dloss = float(grad_alpha @ (eps * d_alpha))
    pred_dloss_from_dx = float(grad_x @ dx)

    prob.params.alpha = alpha0 + eps * d_alpha
    sol_p = solve_one(prob, solver, sid, args.backend)
    act_p = extract_active_set(prob.net, sol_p, tol=st.active_tol, params=prob.params, loss_hat=float(prob.loss_hat[sid]))
    loss_p, _ = dispatch_loss_grad_x(prob.template, sol_p, prob.ds.pg_ac[sid])
    prob.params.alpha = alpha0

    print("\n[One-direction sensitivity / adjoint check]")
    print(f"  moved alpha mass        : bus {j} -> bus {i}")
    print(f"  eps                    : {eps:.3e}")
    print(f"  active set unchanged   : {act.key() == act_p.key()}")
    print(f"  ||dx_fd - dx_kkt||_inf  : {np.linalg.norm((sol_p.x - sol.x) - dx, ord=np.inf):.3e}")
    print(f"  base loss              : {loss0:.8g}")
    print(f"  forward dloss          : {loss_p - loss0:.6e}")
    print(f"  adjoint predicted      : {pred_dloss:.6e}")
    print(f"  grad_x @ dx predicted  : {pred_dloss_from_dx:.6e}")
    print(f"  grad norm / sum        : {np.linalg.norm(grad_alpha):.3e} / {grad_alpha.sum():.3e}")

    print("\nStep 4 passes if the KKT reconstruction error is near the solver tolerance and")
    print("the finite-difference loss change agrees with the adjoint prediction when the active set is unchanged.")


if __name__ == "__main__":
    main()
