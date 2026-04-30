from pathlib import Path
from collections import Counter
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from softopf.config import Settings
from softopf.pipeline import build_problem, make_solver, solve_indices
from softopf.active_set import extract_active_set
from softopf.gradient import batch_loss_and_grads, dispatch_loss_grad_x


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


def batch_loss(problem, solver, idx, backend):
    sols = solve_indices(problem, solver, idx, backend)
    vals = [dispatch_loss_grad_x(problem.template, s, problem.ds.pg_ac[int(i)])[0]
            for i, s in zip(idx, sols)]
    return float(np.mean(vals)), sols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default="data/case39.mat")
    p.add_argument("--case", default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--backend", default="cvxpy", choices=["cvxpy", "osqp"])
    p.add_argument("--solver", default="auto")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--eps", type=float, default=1e-4)
    args = p.parse_args()

    st = Settings(cvxpy_solver=args.solver, active_tol=args.tol)
    prob = build_problem(resolve(args.mat), resolve(args.case), st)
    solver = make_solver(prob, args.backend)

    idx = np.arange(args.offset, min(args.offset + args.batch, len(prob.ds)))
    alpha0 = prob.params.alpha.copy()
    loss0, sols0 = batch_loss(prob, solver, idx, args.backend)
    acts0 = [extract_active_set(prob.net, s, tol=st.active_tol, params=prob.params, loss_hat=float(prob.loss_hat[int(i)])) for i, s in zip(idx, sols0)]
    bg = batch_loss_and_grads(prob, idx, sols0, acts0)

    d_alpha, i, j = feasible_direction(alpha0)
    eps = min(args.eps, 0.25 * alpha0[j])
    prob.params.alpha = alpha0 + eps * d_alpha
    lossp, solsp = batch_loss(prob, solver, idx, args.backend)
    actsp = [extract_active_set(prob.net, s, tol=st.active_tol, params=prob.params, loss_hat=float(prob.loss_hat[int(i)])) for i, s in zip(idx, solsp)]
    prob.params.alpha = alpha0

    unchanged = sum(a.key() == b.key() for a, b in zip(acts0, actsp))
    pred = float(bg.grad_alpha @ (eps * d_alpha))

    print("[Batch reduced-KKT gradient]")
    print(f"  backend                 : {args.backend}")
    print(f"  samples                 : {len(idx)}")
    print(f"  active-set groups       : {bg.n_groups}")
    print(f"  largest group sizes     : {bg.group_sizes[:10]}")
    print(f"  loss direct / grouped   : {loss0:.8g} / {bg.loss:.8g}")
    print(f"  grad norm / sum         : {np.linalg.norm(bg.grad_alpha):.3e} / {bg.grad_alpha.sum():.3e}")
    print(f"  moved alpha mass        : bus {j} -> bus {i}")
    print(f"  eps                    : {eps:.3e}")
    print(f"  unchanged active sets   : {unchanged} / {len(idx)}")
    print(f"  forward dloss           : {lossp - loss0:.6e}")
    print(f"  adjoint predicted       : {pred:.6e}")
    print(f"  status counts           : {dict(Counter(s.status for s in sols0))}")

    print("\nThis is the Step-4 object needed by training: one forward pass gives solutions and")
    print("active sets; then one sparse KKT factorization per active-set group gives the batch alpha-gradient.")


if __name__ == "__main__":
    main()
