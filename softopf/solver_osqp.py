import numpy as np
import osqp
from .params import Params
from .solution import OPFSolution
from .template import SoftQPTemplate


class SoftOPFSolver:
    """Reusable OSQP setup for the fixed-b subcase.

    Use the CVXPY/GUROBI backend for Step-7 b-training. This class is retained
    for quick fixed-b checks and earlier scripts.
    """

    def __init__(self, template: SoftQPTemplate):
        self.template = template
        st = template.settings
        self.prob = osqp.OSQP()
        self.prob.setup(P=template.P, q=template.q, A=template.A,
                        l=template.base_l, u=template.base_u,
                        verbose=False, eps_abs=st.osqp_eps_abs,
                        eps_rel=st.osqp_eps_rel, max_iter=st.osqp_max_iter,
                        polish=st.osqp_polish,
                        scaled_termination=st.osqp_scaled_termination)

    def solve_one(self, pd: np.ndarray, params: Params, loss_hat: float,
                  sample_id: int | None = None) -> OPFSolution:
        if np.linalg.norm(params.b - self.template.net.bphys, np.inf) > 1e-12:
            raise ValueError("OSQP fixed-matrix backend cannot train b; use --backend cvxpy.")
        l, u = self.template.bounds(pd, params, loss_hat)
        self.prob.update(l=l, u=u)
        r = self.prob.solve()
        x = r.x.copy()
        pg, theta, spv, smv = self.template.split(x)
        return OPFSolution(x=x, pg=pg.copy(), theta=theta.copy(),
                           sp=spv.copy(), sm=smv.copy(), y=r.y.copy(),
                           obj=float(r.info.obj_val), status=r.info.status,
                           iter=int(r.info.iter))

    def solve_batch(self, pd_batch: np.ndarray, params: Params,
                    loss_hat_batch: np.ndarray, sample_ids=None):
        return [self.solve_one(pd, params, lh) for pd, lh in zip(pd_batch, loss_hat_batch)]
