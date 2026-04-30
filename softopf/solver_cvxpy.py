import numpy as np
import cvxpy as cp
from .params import Params
from .solution import OPFSolution
from .template import SoftQPTemplate


class CvxpyForwardSolver:
    """Reusable CVXPY model for the Step-7 parameterized QP.

    b appears as a CVXPY Parameter multiplying theta, so the same problem object
    is reused while b, alpha, rbias, gamma, pd, and loss_hat are updated.
    """

    def __init__(self, template: SoftQPTemplate, solver: str = "auto", verbose: bool = False):
        self.template = template
        self.solver = solver
        self.verbose = verbose
        net, v = template.net, template.v

        self.x = cp.Variable(template.nx)
        self.rhs_bal = cp.Parameter(net.n)
        self.line_p_ub = cp.Parameter(net.m)
        self.line_m_ub = cp.Parameter(net.m)
        self.b = cp.Parameter(net.m, nonneg=True)

        pg = self.x[v.pg]
        theta = self.x[v.theta]
        spv = self.x[v.sp]
        smv = self.x[v.sm]
        flow = cp.multiply(self.b, net.Ar @ theta)

        obj = 0.5 * cp.quad_form(self.x, template.P) + template.q @ self.x
        cons = [
            net.Cg @ pg - net.A.T @ flow == self.rhs_bal,
            pg >= net.pg_min,
            pg <= net.pg_max,
            flow - spv <= self.line_p_ub,
            -flow - smv <= self.line_m_ub,
            spv >= 0,
            smv >= 0,
        ]
        self.problem = cp.Problem(cp.Minimize(obj), cons)
        try:
            self.is_dpp = bool(self.problem.is_dpp())
        except Exception:
            self.is_dpp = False

    def _solve_problem(self):
        if self.solver == "GUROBI" or self.solver == "auto":
            try:
                self.problem.solve(solver=cp.GUROBI, verbose=self.verbose, reoptimize=True)
                return "GUROBI"
            except Exception:
                if self.solver == "GUROBI":
                    raise
        if self.solver == "CLARABEL" or self.solver == "auto":
            try:
                self.problem.solve(solver=cp.CLARABEL, verbose=self.verbose)
                return "CLARABEL"
            except Exception:
                if self.solver == "CLARABEL":
                    raise
        self.problem.solve(solver=cp.OSQP, verbose=self.verbose, eps_abs=1e-6, eps_rel=1e-6, max_iter=50000)
        return "OSQP"

    def solve_one(self, pd: np.ndarray, params: Params, loss_hat: float,
                  sample_id: int | None = None) -> OPFSolution:
        l, u = self.template.bounds(pd, params, loss_hat)
        self.rhs_bal.value = u[self.template.rows.bal]
        self.line_p_ub.value = u[self.template.rows.line_p]
        self.line_m_ub.value = u[self.template.rows.line_m]
        self.b.value = params.b
        used = self._solve_problem()
        x = np.asarray(self.x.value, dtype=float).reshape(-1)
        pg, theta, spv, smv = self.template.split(x)
        return OPFSolution(x=x, pg=pg.copy(), theta=theta.copy(),
                           sp=spv.copy(), sm=smv.copy(), y=None,
                           obj=float(self.problem.value),
                           status=f"cvxpy:{used}:{self.problem.status}", iter=0)

    def solve_batch(self, pd_batch: np.ndarray, params: Params,
                    loss_hat_batch: np.ndarray, sample_ids=None):
        return [self.solve_one(pd, params, lh) for pd, lh in zip(pd_batch, loss_hat_batch)]
