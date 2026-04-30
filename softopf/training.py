from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path
import csv
import numpy as np

from .active_set import ActiveSet, extract_active_set
from .region_quad import build_region_quadratic_path
from .gradient import batch_loss_and_grads, dispatch_loss_grad_x
from .params import project_alpha
from .pipeline import solve_indices


@dataclass
class ProposedAdamState:
    t: int
    m: np.ndarray
    v: np.ndarray


@dataclass
class AdamState:
    """Adam only proposes a descent direction; line search decides acceptance."""
    shape: tuple[int, ...]
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self):
        self.t = 0
        self.m = np.zeros(self.shape)
        self.v = np.zeros(self.shape)

    def propose(self, grad: np.ndarray, lr: float) -> tuple[np.ndarray, ProposedAdamState]:
        g = np.asarray(grad, dtype=float).reshape(self.shape)
        t = self.t + 1
        m = self.beta1 * self.m + (1.0 - self.beta1) * g
        v = self.beta2 * self.v + (1.0 - self.beta2) * (g * g)
        mhat = m / (1.0 - self.beta1 ** t)
        vhat = v / (1.0 - self.beta2 ** t)
        d = -lr * mhat / (np.sqrt(vhat) + self.eps)
        if float(g @ d) >= 0.0:
            d = -lr * g
        return d, ProposedAdamState(t=t, m=m, v=v)

    def commit(self, proposed: ProposedAdamState):
        self.t = proposed.t
        self.m = proposed.m.copy()
        self.v = proposed.v.copy()

    def reset(self):
        self.t = 0
        self.m.fill(0.0)
        self.v.fill(0.0)


@dataclass
class ForwardBatch:
    loss: float
    obj: float
    sols: list
    active_sets: list[ActiveSet]
    status_counts: dict


@dataclass
class BlockTrainConfig:
    cycles: int = 5
    phase_iters: int = 40
    min_phase_iters: int = 5
    patience: int = 8
    batch_size: int = 0                  # 0 means full-gradient on train_idx
    optimizer: str = "adam"             # "adam" or "pgd"
    lr_alpha: float = 2e-2
    lr_rbias: float = 0.5
    lr_b: float = 5e-2
    lr_gamma: float = 5e-2
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    backtrack: float = 0.5
    max_trials: int = 14
    armijo_c: float = 1e-4
    stat_tol: float = 1e-9
    min_pred: float = 1e-12
    seed: int = 1
    active_tol: float = 1e-5
    eval_every: int = 10
    # Step caps are L1 caps on the actual projected parameter displacement.
    max_step_l1_alpha: float = 0.15
    max_step_l1_rbias: float = 10.0
    max_step_l1_b: float = 100.0
    max_step_l1_gamma: float = 0.5
    # Parameter boxes/regularization. b is optimized in raw susceptance units.
    b_min_frac: float = 0.5
    b_max_frac: float = 1.5
    gamma_max: float = 0.5
    rbias_reg: float = 1e-4
    b_reg: float = 1e-3
    gamma_reg: float = 1e-4
    tol_abs: float = 1e-1
    tol_rel: float = 1e-5
    cycle_tol_abs: float = 1e-1
    cycle_tol_rel: float = 1e-5
    reset_adam_on_reject: bool = True
    verbose_trials: bool = False
    # Step 10: RHS blocks use per-sample hybrid safe-region quadratic/QP line search.
    # b still uses the true forward-QP line search.
    region_safety: float = 0.995
    critical_margin: float = 1e-8
    critical_dual_tol: float = 1e-8


@dataclass
class TrainRecord:
    global_iter: int
    cycle: int
    phase: str
    phase_iter: int
    train_obj_before: float
    train_obj_after: float
    train_loss_after: float
    batch_loss: float
    trial_obj: float
    val_obj: float
    val_loss: float
    grad_norm: float
    grad_sum: float
    n_groups: int
    accepted: int
    trials: int
    tau: float
    step_l1: float
    step_linf: float
    pred_change: float
    changed_grad_active: int
    eval_mode: str
    qp_evals: int
    tau_region: float
    reason: str
    status_counts: str


@dataclass
class LineSearchResult:
    accepted: bool
    value: np.ndarray
    fb_accept: ForwardBatch | None
    trial: int
    tau: float
    pred: float
    changed_grad_active: int
    trial_obj: float
    reason: str
    eval_mode: str
    qp_evals: int
    tau_region: float


def _b0(problem) -> np.ndarray:
    return problem.net.bphys


def regularized_objective(problem, loss: float, cfg: BlockTrainConfig) -> float:
    p = problem.params
    bscale = np.maximum(np.abs(_b0(problem)), 1e-12)
    bdev = (p.b - _b0(problem)) / bscale
    return float(loss
                 + 0.5 * cfg.rbias_reg * np.dot(p.rbias, p.rbias)
                 + 0.5 * cfg.b_reg * np.dot(bdev, bdev)
                 + 0.5 * cfg.gamma_reg * (np.dot(p.gamma_p, p.gamma_p) + np.dot(p.gamma_m, p.gamma_m)))


def forward_batch(problem, solver, indices, backend: str, active_tol: float,
                  cfg: BlockTrainConfig) -> ForwardBatch:
    idx = np.asarray(indices, dtype=int)
    sols = solve_indices(problem, solver, idx, backend)
    losses = [dispatch_loss_grad_x(problem.template, sol, problem.ds.pg_ac[int(i)])[0]
              for i, sol in zip(idx, sols)]
    active = [extract_active_set(problem.net, sol, tol=active_tol,
                                 params=problem.params, loss_hat=float(problem.loss_hat[int(i)]))
              for i, sol in zip(idx, sols)]
    loss = float(np.mean(losses))
    obj = regularized_objective(problem, loss, cfg)
    return ForwardBatch(loss=loss, obj=obj, sols=sols, active_sets=active,
                        status_counts=dict(Counter(sol.status for sol in sols)))


def get_block(problem, block: str) -> np.ndarray:
    p = problem.params
    if block == "alpha":
        return p.alpha.copy()
    if block == "rbias":
        return p.rbias.copy()
    if block == "b":
        return p.b.copy()
    if block == "gamma":
        return np.r_[p.gamma_p, p.gamma_m]
    raise ValueError(block)


def project_block(problem, block: str, value: np.ndarray, cfg: BlockTrainConfig) -> np.ndarray:
    v = np.asarray(value, dtype=float).reshape(-1)
    if block == "alpha":
        return project_alpha(v)
    if block == "rbias":
        return v                           # Step 10 inherits Step 7: no zero-sum projection.
    if block == "b":
        b0 = _b0(problem)
        return np.clip(v, cfg.b_min_frac * b0, cfg.b_max_frac * b0)
    if block == "gamma":
        return np.clip(v, 0.0, cfg.gamma_max)
    raise ValueError(block)


def set_block(problem, block: str, value: np.ndarray, cfg: BlockTrainConfig):
    v = project_block(problem, block, value, cfg)
    p = problem.params
    if block == "alpha":
        p.alpha = v
    elif block == "rbias":
        p.rbias = v
    elif block == "b":
        p.b = v
    elif block == "gamma":
        m = problem.net.m
        p.gamma_p = v[:m]
        p.gamma_m = v[m:]
    else:
        raise ValueError(block)


def _active_dict(indices, active_sets):
    return {int(i): a for i, a in zip(np.asarray(indices, dtype=int), active_sets)}


def _max_step(cfg: BlockTrainConfig, block: str) -> float:
    return dict(alpha=cfg.max_step_l1_alpha, rbias=cfg.max_step_l1_rbias,
                b=cfg.max_step_l1_b, gamma=cfg.max_step_l1_gamma)[block]


def _lr(cfg: BlockTrainConfig, block: str) -> float:
    return dict(alpha=cfg.lr_alpha, rbias=cfg.lr_rbias, b=cfg.lr_b, gamma=cfg.lr_gamma)[block]


def _cap_direction(problem, block: str, x0: np.ndarray, direction: np.ndarray,
                   cfg: BlockTrainConfig) -> np.ndarray:
    cap = _max_step(cfg, block)
    if cap <= 0.0:
        return direction
    cand = project_block(problem, block, x0 + direction, cfg)
    step_l1 = float(np.linalg.norm(cand - x0, 1))
    return direction if step_l1 <= cap else direction * (cap / max(step_l1, 1e-16))


def _choose_grad_indices(train_idx: np.ndarray, cfg: BlockTrainConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.batch_size <= 0 or cfg.batch_size >= len(train_idx):
        return train_idx
    return rng.choice(train_idx, size=cfg.batch_size, replace=False)


def _block_grad(problem, bg, block: str, cfg: BlockTrainConfig) -> np.ndarray:
    p = problem.params
    if block == "alpha":
        return bg.grad_alpha
    if block == "rbias":
        return bg.grad_rbias + cfg.rbias_reg * p.rbias
    if block == "b":
        bscale = np.maximum(np.abs(_b0(problem)), 1e-12)
        return bg.grad_b + cfg.b_reg * (p.b - _b0(problem)) / (bscale * bscale)
    if block == "gamma":
        return np.r_[bg.grad_gamma_p + cfg.gamma_reg * p.gamma_p,
                     bg.grad_gamma_m + cfg.gamma_reg * p.gamma_m]
    raise ValueError(block)



def _objective_with_candidate(problem, block: str, cand: np.ndarray, loss: float,
                              x0: np.ndarray, cfg: BlockTrainConfig) -> float:
    set_block(problem, block, cand, cfg)
    obj = regularized_objective(problem, loss, cfg)
    set_block(problem, block, x0, cfg)
    return obj


def _rhs_hybrid_line_search(problem, solver, backend: str, block: str,
                            grad_idx: np.ndarray, accept_idx: np.ndarray,
                            x0: np.ndarray, obj0: float, base_fb: ForwardBatch,
                            active_grad0: list[ActiveSet], grad: np.ndarray,
                            full_step: np.ndarray, cfg: BlockTrainConfig) -> LineSearchResult:
    """Hybrid RHS line search: affine inside each sample's safe region, QP outside.

    For alpha/rbias/gamma, each sample has its own certified critical-region
    radius.  At a trial tau, samples satisfying tau <= region_safety*tau_q are
    evaluated by the exact fixed-active-set quadratic loss and no QP is solved.
    Only the remaining samples are re-solved by the true forward QP.  The Armijo
    test is therefore performed on a mixed but valid objective evaluation.
    """
    accept_idx = np.asarray(accept_idx, dtype=int)
    grad_idx = np.asarray(grad_idx, dtype=int)
    path = build_region_quadratic_path(problem, accept_idx, base_fb.active_sets,
                                       block, full_step,
                                       margin=cfg.critical_margin,
                                       dual_tol=cfg.critical_dual_tol)
    finite = path.tau_samples[np.isfinite(path.tau_samples)]
    tau_region_min = float(finite.min()) if finite.size else np.inf

    tau = 1.0
    best = None
    for trial in range(1, cfg.max_trials + 1):
        cand = x0 + tau * full_step
        pred = float(grad @ (tau * full_step))
        if pred >= -cfg.min_pred:
            best = LineSearchResult(False, cand, None, trial, tau, pred, 0,
                                    obj0, "not_descent_direction", "none", 0,
                                    tau_region_min)
            tau *= cfg.backtrack
            continue

        safe = tau <= cfg.region_safety * path.tau_samples
        unsafe_loc = np.flatnonzero(~safe)
        unsafe_ids = accept_idx[unsafe_loc]

        # Safe samples: exact quadratic loss and affine solution in the current
        # active set.  Unsafe samples: true forward QP at the candidate point.
        sample_losses = path.sample_loss(tau)
        new_sols = [None] * len(accept_idx)
        new_active = [None] * len(accept_idx)
        for loc in np.flatnonzero(safe):
            new_sols[int(loc)] = path.sol(problem.template, int(loc), tau)
            new_active[int(loc)] = base_fb.active_sets[int(loc)]

        status_counts = Counter()
        status_counts["fixed_active_quad"] = int(np.sum(safe))
        qp_sample_count = int(len(unsafe_ids))
        if qp_sample_count > 0:
            set_block(problem, block, cand, cfg)
            qsols = solve_indices(problem, solver, unsafe_ids, backend)

            # Unsafe samples are solved at the candidate parameters, so their
            # active sets must also be extracted under the candidate parameters.
            # Reset to x0 only after active-set extraction is finished.
            for j, loc in enumerate(unsafe_loc):
                sid = int(unsafe_ids[j])
                sol = qsols[j]
                ell, _ = dispatch_loss_grad_x(problem.template, sol, problem.ds.pg_ac[sid])
                sample_losses[int(loc)] = ell
                new_sols[int(loc)] = sol
                new_active[int(loc)] = extract_active_set(
                    problem.net, sol, tol=cfg.active_tol,
                    params=problem.params, loss_hat=problem.loss_hat[sid]
                )
                status_counts[sol.status] += 1

            set_block(problem, block, x0, cfg)

        trial_loss = float(np.mean(sample_losses))
        trial_obj = _objective_with_candidate(problem, block, cand, trial_loss, x0, cfg)
        rhs = obj0 + cfg.armijo_c * pred + cfg.stat_tol
        ok = trial_obj <= rhs

        cand_active = {int(sid): act for sid, act in zip(accept_idx, new_active)}
        changed = sum(1 for sid, old in zip(grad_idx, active_grad0)
                      if int(sid) in cand_active and old.key() != cand_active[int(sid)].key())
        reason = "accepted_hybrid" if ok else "hybrid_no_armijo_decrease"
        mode = "hybrid" if qp_sample_count > 0 else "quad"
        fb = None
        if ok:
            fb = ForwardBatch(loss=trial_loss, obj=trial_obj, sols=new_sols,
                              active_sets=new_active, status_counts=dict(status_counts))
        best = LineSearchResult(ok, cand, fb, trial, tau, pred, changed,
                                trial_obj, reason, mode, qp_sample_count,
                                tau_region_min)
        if cfg.verbose_trials:
            print(f"    {block} trial={trial:02d} tau={tau:.2e} mode={mode} obj={trial_obj:.8g} "
                  f"rhs={rhs:.8g} pred={pred:.3e} safe={int(np.sum(safe))}/{len(accept_idx)} "
                  f"qp_samples={qp_sample_count} changed={changed}/{len(grad_idx)} {reason}")
        if ok:
            return best
        tau *= cfg.backtrack

    set_block(problem, block, x0, cfg)
    return best if best is not None else LineSearchResult(False, x0, None, cfg.max_trials,
                                                          0.0, 0.0, 0, obj0,
                                                          "no_hybrid_trial", "none", 0,
                                                          tau_region_min)


def _qp_line_search(problem, solver, backend: str, block: str,
                    grad_idx: np.ndarray, accept_idx: np.ndarray,
                    x0: np.ndarray, obj0: float, active_grad0: list[ActiveSet],
                    grad: np.ndarray, full_step: np.ndarray,
                    cfg: BlockTrainConfig) -> LineSearchResult:
    """True forward-QP Armijo search, used for b."""
    tau = 1.0
    best = None
    qp_evals = 0
    for trial in range(1, cfg.max_trials + 1):
        cand = x0 + tau * full_step
        pred = float(grad @ (cand - x0))
        if pred >= -cfg.min_pred:
            best = LineSearchResult(False, cand, None, trial, tau, pred, 0, obj0,
                                    "not_descent_direction", "none", qp_evals, np.nan)
            tau *= cfg.backtrack
            continue

        set_block(problem, block, cand, cfg)
        fb = forward_batch(problem, solver, accept_idx, backend, cfg.active_tol, cfg)
        qp_evals += 1
        cand_active = _active_dict(accept_idx, fb.active_sets)
        changed = sum(1 for sid, old in zip(grad_idx, active_grad0)
                      if int(sid) in cand_active and old.key() != cand_active[int(sid)].key())
        rhs = obj0 + cfg.armijo_c * pred + cfg.stat_tol
        ok = fb.obj <= rhs
        reason = "accepted_qp" if ok else "qp_no_armijo_decrease"
        best = LineSearchResult(ok, cand, fb if ok else None, trial, tau, pred,
                                changed, fb.obj, reason, "qp", qp_evals, np.nan)
        if cfg.verbose_trials:
            print(f"    {block} trial={trial:02d} tau={tau:.2e} mode=qp obj={fb.obj:.8g} "
                  f"rhs={rhs:.8g} pred={pred:.3e} changed={changed}/{len(grad_idx)} {reason}")
        if ok:
            return best
        tau *= cfg.backtrack

    set_block(problem, block, x0, cfg)
    return best if best is not None else LineSearchResult(False, x0, None, cfg.max_trials,
                                                          0.0, 0.0, 0, obj0,
                                                          "no_qp_trial", "none", qp_evals,
                                                          np.nan)


def line_search_block(problem, solver, backend: str, block: str,
                      grad_idx: np.ndarray, accept_idx: np.ndarray,
                      x0: np.ndarray, obj0: float, base_fb: ForwardBatch,
                      active_grad0: list[ActiveSet], grad: np.ndarray,
                      direction: np.ndarray, cfg: BlockTrainConfig) -> LineSearchResult:
    """Line search: RHS blocks use per-sample hybrid quadratic/QP evaluation; b uses true QP."""
    end_point = project_block(problem, block, x0 + direction, cfg)
    full_step = end_point - x0
    if np.linalg.norm(full_step, 1) <= 0.0:
        return LineSearchResult(False, x0, None, 0, 0.0, 0.0, 0, obj0,
                                "zero_projected_step", "none", 0, np.nan)

    if block in {"alpha", "rbias", "gamma"}:
        return _rhs_hybrid_line_search(problem, solver, backend, block, grad_idx, accept_idx, x0,
                                       obj0, base_fb, active_grad0, grad, full_step, cfg)
    return _qp_line_search(problem, solver, backend, block, grad_idx, accept_idx,
                           x0, obj0, active_grad0, grad, full_step, cfg)
def evaluate_params(problem, solver, indices, backend: str, cfg: BlockTrainConfig) -> dict:
    fb = forward_batch(problem, solver, indices, backend, cfg.active_tol, cfg)
    return {"loss": fb.loss, "obj": fb.obj, "status_counts": fb.status_counts}


def train_one_block(problem, solver, train_idx: np.ndarray, val_idx: np.ndarray | None,
                    backend: str, block: str, cfg: BlockTrainConfig,
                    opt: AdamState, rng: np.random.Generator,
                    current_fb: ForwardBatch, cycle: int, global_iter0: int,
                    history: list[TrainRecord]) -> tuple[ForwardBatch, bool, int]:
    no_improve = 0
    converged = False
    global_iter = global_iter0

    for pit in range(1, cfg.phase_iters + 1):
        global_iter += 1
        grad_idx = _choose_grad_indices(train_idx, cfg, rng)
        x0 = get_block(problem, block)
        obj0 = current_fb.obj

        fb_grad = forward_batch(problem, solver, grad_idx, backend, cfg.active_tol, cfg)
        bg = batch_loss_and_grads(problem, grad_idx, fb_grad.sols, fb_grad.active_sets)
        grad = _block_grad(problem, bg, block, cfg)

        if cfg.optimizer == "pgd":
            direction = -_lr(cfg, block) * grad
            proposed = None
        else:
            direction, proposed = opt.propose(grad, _lr(cfg, block))
        direction = _cap_direction(problem, block, x0, direction, cfg)

        ls = line_search_block(problem, solver, backend, block, grad_idx, train_idx,
                               x0, obj0, current_fb, fb_grad.active_sets,
                               grad, direction, cfg)
        if ls.accepted:
            set_block(problem, block, ls.value, cfg)
            if proposed is not None:
                opt.commit(proposed)
            current_fb = ls.fb_accept
            step = get_block(problem, block) - x0
            improvement = obj0 - current_fb.obj
            threshold = cfg.tol_abs + cfg.tol_rel * max(1.0, abs(obj0))
            no_improve = 0 if improvement > threshold else no_improve + 1
            accepted = 1
        else:
            set_block(problem, block, x0, cfg)
            if cfg.reset_adam_on_reject:
                opt.reset()
            step = np.zeros_like(x0)
            improvement = 0.0
            no_improve += 1
            accepted = 0

        val_obj = np.nan
        val_loss = np.nan
        if val_idx is not None and cfg.eval_every > 0 and (global_iter % cfg.eval_every == 0):
            ev = evaluate_params(problem, solver, val_idx, backend, cfg)
            val_obj, val_loss = float(ev["obj"]), float(ev["loss"])

        rec = TrainRecord(
            global_iter=global_iter, cycle=cycle, phase=block, phase_iter=pit,
            train_obj_before=float(obj0), train_obj_after=float(current_fb.obj),
            train_loss_after=float(current_fb.loss), batch_loss=float(fb_grad.loss),
            trial_obj=float(ls.trial_obj), val_obj=float(val_obj), val_loss=float(val_loss),
            grad_norm=float(np.linalg.norm(grad)), grad_sum=float(grad.sum()),
            n_groups=bg.n_groups, accepted=accepted, trials=int(ls.trial), tau=float(ls.tau),
            step_l1=float(np.linalg.norm(step, 1)), step_linf=float(np.linalg.norm(step, np.inf)),
            pred_change=float(ls.pred), changed_grad_active=int(ls.changed_grad_active),
            eval_mode=ls.eval_mode, qp_evals=int(ls.qp_evals), tau_region=float(ls.tau_region),
            reason=ls.reason, status_counts=str({} if ls.fb_accept is None else ls.fb_accept.status_counts))
        history.append(rec)

        msg = (f"cy={cycle:02d} {block:6s} it={pit:03d} "
               f"obj={rec.train_obj_before:.6g}->{rec.train_obj_after:.6g} "
               f"loss={rec.train_loss_after:.6g} batch={rec.batch_loss:.6g} "
               f"|g|={rec.grad_norm:.3e} groups={rec.n_groups:2d} acc={accepted} "
               f"trials={rec.trials} tau={rec.tau:.2e} step1={rec.step_l1:.2e} "
               f"mode={rec.eval_mode} qp={rec.qp_evals} region={rec.tau_region:.2e} "
               f"changed={rec.changed_grad_active}/{len(grad_idx)} reason={rec.reason}")
        if not np.isnan(val_obj):
            msg += f" val_obj={val_obj:.6g}"
        print(msg)

        if pit >= cfg.min_phase_iters and no_improve >= cfg.patience:
            converged = True
            break

    return current_fb, converged, global_iter


def train_step10(problem, solver, train_idx, val_idx=None, backend: str = "cvxpy",
                cfg: BlockTrainConfig | None = None, out_dir: str | Path | None = None):
    """Block-coordinate training: alpha, rbias, b, gamma.

    Compared with Step 7, Step 10 avoids unnecessary forward-QP solves for RHS-block line search.
    For alpha/rbias/gamma, it constructs fixed-active-set affine solution maps sample by sample. During line search, safe samples are evaluated by the exact quadratic loss, while unsafe samples are re-solved by the true forward QP. The b block still uses true forward-QP line search.
    """
    cfg = cfg or BlockTrainConfig()
    rng = np.random.default_rng(cfg.seed)
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = None if val_idx is None else np.asarray(val_idx, dtype=int)
    blocks = ["alpha", "rbias", "b", "gamma"]
    opts = {b: AdamState(get_block(problem, b).shape, cfg.beta1, cfg.beta2, cfg.adam_eps)
            for b in blocks}
    history: list[TrainRecord] = []
    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    current_fb = forward_batch(problem, solver, train_idx, backend, cfg.active_tol, cfg)
    print(f"initial train obj/loss : {current_fb.obj:.8g} / {current_fb.loss:.8g}")
    if val_idx is not None:
        ev = evaluate_params(problem, solver, val_idx, backend, cfg)
        print(f"initial val   obj/loss : {ev['obj']:.8g} / {ev['loss']:.8g}")

    global_iter = 0
    for cycle in range(1, cfg.cycles + 1):
        cycle_start = current_fb.obj
        gains = {}
        conv = {}
        for block in blocks:
            print(f"\n[Cycle {cycle}: {block} phase]")
            before = current_fb.obj
            current_fb, conv[block], global_iter = train_one_block(
                problem, solver, train_idx, val_idx, backend, block, cfg,
                opts[block], rng, current_fb, cycle, global_iter, history)
            gains[block] = before - current_fb.obj

        total_gain = cycle_start - current_fb.obj
        stop_thr = cfg.cycle_tol_abs + cfg.cycle_tol_rel * max(1.0, abs(cycle_start))
        print(f"\n[Cycle {cycle} summary] obj {cycle_start:.8g} -> {current_fb.obj:.8g}; "
              + ", ".join(f"{k}_gain={v:.3e}" for k, v in gains.items())
              + f", total_gain={total_gain:.3e}")
        if all(conv.values()) and total_gain <= stop_thr:
            print("[Step-10 hybrid line-search training converged]")
            break

    if out_path is not None:
        p = problem.params
        np.save(out_path / "alpha.npy", p.alpha)
        np.save(out_path / "rbias.npy", p.rbias)
        np.save(out_path / "b.npy", p.b)
        np.save(out_path / "gamma_p.npy", p.gamma_p)
        np.save(out_path / "gamma_m.npy", p.gamma_m)
        np.savez(out_path / "params_step10.npz", alpha=p.alpha, rbias=p.rbias,
                 b=p.b, gamma_p=p.gamma_p, gamma_m=p.gamma_m)
        if history:
            with open(out_path / "train_log.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(history[0]).keys()))
                writer.writeheader()
                for r in history:
                    writer.writerow(asdict(r))
    return problem.params, history


# Compatibility wrappers.
def train_step7(problem, solver, train_idx, val_idx=None, backend: str = "cvxpy",
                cfg: BlockTrainConfig | None = None, out_dir: str | Path | None = None):
    return train_step10(problem, solver, train_idx, val_idx, backend, cfg, out_dir)


def train_alpha_rbias(problem, solver, train_idx, val_idx=None, backend: str = "cvxpy",
                      cfg: BlockTrainConfig | None = None, out_dir: str | Path | None = None):
    return train_step10(problem, solver, train_idx, val_idx, backend, cfg, out_dir)


def train_alpha(problem, solver, train_idx, val_idx=None, backend: str = "cvxpy",
                cfg: BlockTrainConfig | None = None, out_dir: str | Path | None = None):
    return train_step10(problem, solver, train_idx, val_idx, backend, cfg, out_dir)
