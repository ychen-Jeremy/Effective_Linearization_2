from dataclasses import dataclass


@dataclass
class Settings:
    """Numerical choices for the prototype."""

    # Lower-level softened OPF penalties.
    rho_s: float = 1e4
    eps_s: float = 1e-6

    # Offline loss estimator.
    loss_ridge: float = 1e-6

    # Active-set extraction.
    active_tol: float = 1e-5

    # CVXPY reference solver. auto tries GUROBI, then CLARABEL, then OSQP.
    cvxpy_solver: str = "auto"
    cvxpy_verbose: bool = False

    # Sparse OSQP path. In Step 7, the OSQP path is intended only for the
    # fixed-b case; use CVXPY/GUROBI for b-training.
    osqp_eps_abs: float = 1e-5
    osqp_eps_rel: float = 1e-5
    osqp_max_iter: int = 50000
    osqp_polish: bool = True
    osqp_scaled_termination: bool = True
