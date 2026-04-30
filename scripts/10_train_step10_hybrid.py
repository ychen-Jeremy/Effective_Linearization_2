import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from softopf.config import Settings
from softopf.pipeline import build_problem, make_solver
from softopf.training import BlockTrainConfig, train_step10, evaluate_params
from softopf.objective import dispatch_mse


def _split_indices(n_samples: int, n_train: int, n_val: int, seed: int):
    rng = np.random.default_rng(seed)
    all_idx = rng.permutation(n_samples)
    n_train = min(n_train, n_samples)
    n_val = min(n_val, max(0, n_samples - n_train))
    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val] if n_val > 0 else None
    return train_idx, val_idx


def _print_dc_baseline(prob, train_idx, val_idx):
    dc_train_loss = dispatch_mse(prob.ds.pg_dc[train_idx], prob.ds.pg_ac[train_idx])
    print("\n[Stored vanilla DCOPF baseline]")
    print(f"  train DC loss         : {dc_train_loss:.8g}")
    print(f"  train DC L2/sample    : {np.sqrt(dc_train_loss):.6g} MW")
    print(f"  train DC RMSE/gen     : {np.sqrt(dc_train_loss / prob.net.g):.6g} MW")

    if val_idx is not None:
        dc_val_loss = dispatch_mse(prob.ds.pg_dc[val_idx], prob.ds.pg_ac[val_idx])
        print(f"  val DC loss           : {dc_val_loss:.8g}")
        print(f"  val DC L2/sample      : {np.sqrt(dc_val_loss):.6g} MW")
        print(f"  val DC RMSE/gen       : {np.sqrt(dc_val_loss / prob.net.g):.6g} MW")


def main():
    p = argparse.ArgumentParser(
        description="Step 10: Step-7 block training with per-sample hybrid quadratic/QP line search for RHS blocks."
    )

    # Keep the same interface style as scripts/07_train_step7.py.
    p.add_argument("--mat", default="E:/Research/Model/soft_opf_alpha_step10_hybrid/data/case39.mat")
    p.add_argument("--case", default=None)
    p.add_argument("--backend", choices=["cvxpy", "osqp"], default="cvxpy")
    p.add_argument("--solver", default="auto", choices=["auto", "GUROBI", "CLARABEL", "OSQP"])
    p.add_argument("--n-train", "--train", dest="n_train", type=int, default=32)
    p.add_argument("--n-val", "--val", dest="n_val", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cycles", type=int, default=40)
    p.add_argument("--phase-iters", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=0,
                   help="0 means full gradient")
    p.add_argument("--optimizer", choices=["adam", "pgd"], default="adam")

    # Block learning rates.  b remains the Step-7 raw-susceptance update.
    p.add_argument("--lr-alpha", type=float, default=1e-1)
    p.add_argument("--lr-rbias", type=float, default=1e0)
    p.add_argument("--lr-b", type=float, default=1e3)
    p.add_argument("--lr-gamma", type=float, default=1e-1)

    # Regularization and boxes.
    p.add_argument("--b-reg", type=float, default=1e2)
    p.add_argument("--rbias-reg", type=float, default=1e-2)
    p.add_argument("--gamma-reg", type=float, default=1e-2)
    p.add_argument("--b-min-frac", type=float, default=0.5)
    p.add_argument("--b-max-frac", type=float, default=1.5)
    p.add_argument("--gamma-max", type=float, default=0.5)

    # Step caps are L1 caps on the actual projected parameter displacement.
    p.add_argument("--max-step-l1-alpha", type=float, default=0.15)
    p.add_argument("--max-step-l1-rbias", type=float, default=10.0)
    p.add_argument("--max-step-l1-b", type=float, default=100.0)
    p.add_argument("--max-step-l1-gamma", type=float, default=0.5)

    # Line search / optimizer details.
    p.add_argument("--max-trials", type=int, default=10)
    p.add_argument("--backtrack", type=float, default=0.5)
    p.add_argument("--armijo-c", type=float, default=1e-4)
    p.add_argument("--stat-tol", type=float, default=1e-9)
    p.add_argument("--min-pred", type=float, default=1e-12)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--tol-abs", type=float, default=1e-5)
    p.add_argument("--tol-rel", type=float, default=1e-5)
    p.add_argument("--cycle-tol-abs", type=float, default=1e-4)
    p.add_argument("--cycle-tol-rel", type=float, default=1e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--no-reset-adam-on-reject", action="store_true")

    # Step-10 only: RHS blocks use per-sample safe-region quadratic/QP line search.
    p.add_argument("--region-safety", type=float, default=0.995)
    p.add_argument("--critical-margin", type=float, default=1e-8)
    p.add_argument("--critical-dual-tol", type=float, default=1e-8)

    p.add_argument("--verbose-trials", action="store_true")
    p.add_argument("--out", default="outputs/step10_quadls")
    args = p.parse_args()

    st = Settings(cvxpy_solver=args.solver)
    prob = build_problem(args.mat, args.case, st)
    solver = make_solver(prob, args.backend)

    train_idx, val_idx = _split_indices(len(prob.ds), args.n_train, args.n_val, args.seed)
    _print_dc_baseline(prob, train_idx, val_idx)

    cfg = BlockTrainConfig(
        cycles=args.cycles,
        phase_iters=args.phase_iters,
        patience=args.patience,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        lr_alpha=args.lr_alpha,
        lr_rbias=args.lr_rbias,
        lr_b=args.lr_b,
        lr_gamma=args.lr_gamma,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        backtrack=args.backtrack,
        max_trials=args.max_trials,
        armijo_c=args.armijo_c,
        stat_tol=args.stat_tol,
        min_pred=args.min_pred,
        seed=args.seed,
        active_tol=st.active_tol,
        eval_every=args.eval_every,
        max_step_l1_alpha=args.max_step_l1_alpha,
        max_step_l1_rbias=args.max_step_l1_rbias,
        max_step_l1_b=args.max_step_l1_b,
        max_step_l1_gamma=args.max_step_l1_gamma,
        b_min_frac=args.b_min_frac,
        b_max_frac=args.b_max_frac,
        gamma_max=args.gamma_max,
        rbias_reg=args.rbias_reg,
        b_reg=args.b_reg,
        gamma_reg=args.gamma_reg,
        tol_abs=args.tol_abs,
        tol_rel=args.tol_rel,
        cycle_tol_abs=args.cycle_tol_abs,
        cycle_tol_rel=args.cycle_tol_rel,
        reset_adam_on_reject=not args.no_reset_adam_on_reject,
        verbose_trials=args.verbose_trials,
        region_safety=args.region_safety,
        critical_margin=args.critical_margin,
        critical_dual_tol=args.critical_dual_tol,
    )

    print("\n[Step 10 hybrid setup]")
    print(f"  train / val samples   : {len(train_idx)} / {0 if val_idx is None else len(val_idx)}")
    print(f"  backend / solver      : {args.backend} / {args.solver}")
    print(f"  batch size            : {'full' if cfg.batch_size <= 0 else cfg.batch_size}")
    print(f"  blocks                : alpha -> rbias -> b -> gamma")
    print(f"  b line search         : true forward QP, same as Step 7")
    print(f"  RHS line search       : safe samples use affine quadratic; unsafe samples use full QP")

    train_step10(prob, solver, train_idx, val_idx, args.backend, cfg, ROOT / args.out)

    tr = evaluate_params(prob, solver, train_idx, args.backend, cfg)
    print("\n[Final train]")
    print(f"  obj/loss/status       : {tr['obj']:.8g} / {tr['loss']:.8g} / {tr['status_counts']}")
    if val_idx is not None:
        va = evaluate_params(prob, solver, val_idx, args.backend, cfg)
        print("[Final validation]")
        print(f"  obj/loss/status       : {va['obj']:.8g} / {va['loss']:.8g} / {va['status_counts']}")

    pms = prob.params
    print("\n[Final parameters]")
    print(f"  alpha sum/min/max     : {pms.alpha.sum():.8g} / {pms.alpha.min():.3e} / {pms.alpha.max():.3e}")
    print(f"  rbias sum/norm        : {pms.rbias.sum():.6g} / {np.linalg.norm(pms.rbias):.6g}")
    print(f"  b rel min/max         : {(pms.b / prob.net.bphys).min():.6g} / {(pms.b / prob.net.bphys).max():.6g}")
    print(f"  b rel dev norm/max    : {np.linalg.norm((pms.b - prob.net.bphys) / prob.net.bphys):.6g} / "
          f"{np.max(np.abs((pms.b - prob.net.bphys) / prob.net.bphys)):.6g}")
    print(f"  gamma_p norm/max      : {np.linalg.norm(pms.gamma_p):.6g} / {pms.gamma_p.max():.6g}")
    print(f"  gamma_m norm/max      : {np.linalg.norm(pms.gamma_m):.6g} / {pms.gamma_m.max():.6g}")
    print(f"  saved to              : {ROOT / args.out}")


if __name__ == "__main__":
    main()
