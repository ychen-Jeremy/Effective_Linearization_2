from pathlib import Path
from collections import Counter
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def resolve(path):
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


from softopf.config import Settings
from softopf.pipeline import build_problem, make_solver, solve_one, solve_indices
from softopf.diagnostics import residuals
from softopf.objective import objective_parts, batch_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default="data/case39.mat")
    p.add_argument("--case", default=None)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--backend", default="cvxpy", choices=["cvxpy", "osqp"])
    p.add_argument("--solver", default="auto", help="CVXPY solver name or auto")
    args = p.parse_args()

    st = Settings(cvxpy_solver=args.solver)
    prob = build_problem(resolve(args.mat), resolve(args.case), st)
    solver = make_solver(prob, args.backend)

    sol = solve_one(prob, solver, args.sample, args.backend)
    r = residuals(prob.net, prob.template, prob.ds.pd[args.sample], prob.params,
                  prob.loss_hat[args.sample], sol)
    obj = objective_parts(prob.template, sol)

    print("[Single-sample forward]")
    if args.backend == "cvxpy" and hasattr(solver, "is_dpp"):
        print(f"  cvxpy DPP              : {solver.is_dpp}")
    print(f"  backend / status       : {args.backend} / {sol.status}")
    print(f"  objective             : {sol.obj:.8g}")
    print(f"  objective check       : {obj['total']:.8g}")
    print(f"  balance_inf           : {r['balance_inf']:.3e}")
    print(f"  system balance        : {r['system_balance']:.3e} MW")
    print(f"  gen / line violation  : {r['gen_violation']:.3e} / {r['line_violation']:.3e}")
    print(f"  fixed loss_hat        : {prob.loss_hat[args.sample]:.4f} MW")
    print(f"  slack positive sum    : {r['slack_sum_pos']:.3e}")
    print(f"  max line loading      : {r['max_loading']:.4f}")
    print(f"  Pg first 5            : {np.round(sol.pg[:5], 4)}")
    print(f"  PgAC first 5          : {np.round(prob.ds.pg_ac[args.sample, :5], 4)}")
    print(f"  PgDC first 5          : {np.round(prob.ds.pg_dc[args.sample, :5], 4)}")

    idx = np.arange(min(args.batch, len(prob.ds)))
    sols = solve_indices(prob, solver, idx, args.backend)
    m = batch_metrics(sols, prob.ds.pg_ac[idx])
    max_bal = max(residuals(prob.net, prob.template, prob.ds.pd[i], prob.params,
                            prob.loss_hat[i], s)["balance_inf"] for i, s in zip(idx, sols))
    print("\n[Batch forward]")
    print(f"  batch size            : {len(idx)}")
    print(f"  statuses              : {dict(Counter(m['statuses']))}")
    print(f"  max balance_inf       : {max_bal:.3e}")
    print(f"  dispatch MSE          : {m['dispatch_mse']:.6g}")
    print(f"  mean slack sum        : {m['mean_slack']:.3e}")


if __name__ == "__main__":
    main()
