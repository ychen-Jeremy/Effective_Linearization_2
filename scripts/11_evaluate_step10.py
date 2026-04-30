import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from softopf.config import Settings
from softopf.pipeline import build_problem, make_solver
from softopf.training import BlockTrainConfig, evaluate_params


def main():
    p = argparse.ArgumentParser(description="Evaluate Step-10 learned parameters.")
    p.add_argument("--mat", default="data/case39.mat")
    p.add_argument("--case", default=None)
    p.add_argument("--backend", choices=["cvxpy", "osqp"], default="cvxpy")
    p.add_argument("--solver", default="auto", choices=["auto", "GUROBI", "CLARABEL", "OSQP"])
    p.add_argument("--params", default="outputs/step10_quadls/params_step10.npz")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--seed", type=int, default=2)
    args = p.parse_args()

    st = Settings(cvxpy_solver=args.solver)
    prob = build_problem(args.mat, args.case, st)
    par_path = ROOT / args.params
    z = np.load(par_path)
    prob.params.alpha = z["alpha"]
    prob.params.rbias = z["rbias"]
    prob.params.b = z["b"]
    prob.params.gamma_p = z["gamma_p"]
    prob.params.gamma_m = z["gamma_m"]

    solver = make_solver(prob, args.backend)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(prob.ds))[:min(args.n, len(prob.ds))]
    ev = evaluate_params(prob, solver, idx, args.backend, BlockTrainConfig(active_tol=st.active_tol))

    print("[Step 10 evaluation]")
    print(f"  params                : {par_path}")
    print(f"  samples               : {len(idx)}")
    print(f"  dispatch loss         : {ev['loss']:.8g}")
    print(f"  regularized objective : {ev['obj']:.8g}")
    print(f"  status counts         : {ev['status_counts']}")


if __name__ == "__main__":
    main()
