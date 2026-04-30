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
from softopf.pipeline import build_problem, make_solver, solve_indices
from softopf.active_set import (
    extract_active_set, group_active_sets, representative_counts,
    active_kkt_equalities, active_residuals,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default="data/case39.mat")
    p.add_argument("--case", default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--backend", default="cvxpy", choices=["cvxpy", "osqp"])
    p.add_argument("--solver", default="auto")
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    st = Settings(cvxpy_solver=args.solver, active_tol=args.tol)
    prob = build_problem(resolve(args.mat), resolve(args.case), st)
    solver = make_solver(prob, args.backend)

    idx = np.arange(args.offset, min(args.offset + args.batch, len(prob.ds)))
    sols = solve_indices(prob, solver, idx, args.backend)
    acts = [extract_active_set(prob.net, s, tol=st.active_tol, params=prob.params, loss_hat=float(prob.loss_hat[int(i)])) for i, s in zip(idx, sols)]
    groups = group_active_sets(acts)
    group_sizes = sorted((len(v) for v in groups.values()), reverse=True)

    print("[Active-set summary]")
    print(f"  samples checked        : {len(idx)}")
    print(f"  unique active sets     : {len(groups)}")
    print(f"  largest group sizes    : {group_sizes[:10]}")
    print("  representative groups  :")
    for k, d in enumerate(representative_counts(acts)[:8], start=1):
        print(f"    {k:02d}: size={d.pop('size'):>3d}, {d}")

    rep_local = next(iter(groups.values()))[0]
    sid = int(idx[rep_local])
    a = acts[rep_local]
    G, h, labels = active_kkt_equalities(prob.template, a, prob.ds.pd[sid],
                                         prob.params, prob.loss_hat[sid])
    ar = active_residuals(prob.net, sols[rep_local], a, params=prob.params, loss_hat=float(prob.loss_hat[sid]))
    label_counts = dict(Counter(labels))
    print("\n[Representative reduced-KKT rows]")
    print(f"  sample id              : {sid}")
    print(f"  G_A shape / nnz        : {G.shape} / {G.nnz}")
    print(f"  label counts           : {label_counts}")
    print(f"  active residual max    : {max(ar.values()):.3e}")
    print(f"  residuals by block     : {ar}")

    print("\nNode 3 passes if active residuals are near the forward-solve tolerance and")
    print("the grouped active-set signatures are stable enough to justify reuse in Step 4.")


if __name__ == "__main__":
    main()
