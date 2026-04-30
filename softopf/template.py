from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from .network import Network
from .config import Settings
from .params import Params


@dataclass(frozen=True)
class VarSlices:
    pg: slice
    theta: slice
    sp: slice
    sm: slice


@dataclass(frozen=True)
class RowSlices:
    bal: slice
    pg: slice
    line_p: slice
    line_m: slice
    sp: slice
    sm: slice


class SoftQPTemplate:
    """Sparse QP template for the fixed-loss softened OPF.

    Variables: x = [Pg, theta_nonref, s_plus, s_minus].
    In Step 7, b changes the equality/line-flow matrix, while alpha, rbias,
    gamma_p, and gamma_m change only the RHS bounds.
    """

    def __init__(self, net: Network, settings: Settings):
        self.net = net
        self.settings = settings
        g, n1, m = net.g, net.n - 1, net.m
        self.v = VarSlices(slice(0, g), slice(g, g + n1),
                           slice(g + n1, g + n1 + m), slice(g + n1 + m, g + n1 + 2 * m))
        self.nx = g + n1 + 2 * m
        self.rows = RowSlices(slice(0, net.n), slice(net.n, net.n + g),
                              slice(net.n + g, net.n + g + m),
                              slice(net.n + g + m, net.n + g + 2 * m),
                              slice(net.n + g + 2 * m, net.n + g + 3 * m),
                              slice(net.n + g + 3 * m, net.n + g + 4 * m))
        self.nc = net.n + net.g + 4 * net.m
        self.P = self._build_P()
        self.q = self._build_q()
        self.A = self.build_A(net.bphys)
        self.base_l, self.base_u = self._base_bounds()

    def _build_P(self):
        net, st = self.net, self.settings
        diag = np.zeros(self.nx)
        diag[self.v.pg] = 2.0 * net.c2
        diag[self.v.sp] = st.eps_s
        diag[self.v.sm] = st.eps_s
        return sp.diags(diag, format="csc")

    def _build_q(self):
        net, st = self.net, self.settings
        q = np.zeros(self.nx)
        q[self.v.pg] = net.c1
        q[self.v.sp] = st.rho_s
        q[self.v.sm] = st.rho_s
        return q

    def build_flow_matrix(self, b: np.ndarray):
        """Return Bf(b)=diag(b) A_r."""
        return sp.diags(np.asarray(b, dtype=float).reshape(-1), format="csr") @ self.net.Ar

    def build_bus_matrix(self, b: np.ndarray):
        """Return A' diag(b) A_r."""
        return self.net.A.T @ self.build_flow_matrix(b)

    def build_A(self, b: np.ndarray):
        net = self.net
        Bf = self.build_flow_matrix(b)
        Bbus = net.A.T @ Bf
        blocks = []
        # Nodal balance: Cg Pg - A' diag(b) A_r theta = Pd + loss_hat * alpha + rbias.
        blocks.append(sp.hstack([net.Cg, -Bbus,
                                 sp.csr_matrix((net.n, net.m)),
                                 sp.csr_matrix((net.n, net.m))]))
        # Generator bounds: Pg in [Pmin, Pmax].
        blocks.append(sp.hstack([sp.eye(net.g, format="csr"),
                                 sp.csr_matrix((net.g, net.n - 1 + 2 * net.m))]))
        # Positive line limit: Bf theta - sp <= fmax - gamma_p loss_hat.
        blocks.append(sp.hstack([sp.csr_matrix((net.m, net.g)), Bf,
                                 -sp.eye(net.m, format="csr"),
                                 sp.csr_matrix((net.m, net.m))]))
        # Negative line limit: -Bf theta - sm <= fmax - gamma_m loss_hat.
        blocks.append(sp.hstack([sp.csr_matrix((net.m, net.g)), -Bf,
                                 sp.csr_matrix((net.m, net.m)),
                                 -sp.eye(net.m, format="csr")]))
        # Slack nonnegativity: sp >= 0, sm >= 0.
        blocks.append(sp.hstack([sp.csr_matrix((net.m, net.g + net.n - 1)),
                                 sp.eye(net.m, format="csr"),
                                 sp.csr_matrix((net.m, net.m))]))
        blocks.append(sp.hstack([sp.csr_matrix((net.m, net.g + net.n - 1 + net.m)),
                                 sp.eye(net.m, format="csr")]))
        return sp.vstack(blocks, format="csc")

    def _base_bounds(self):
        net = self.net
        inf = np.inf
        l = -inf * np.ones(self.nc)
        u = inf * np.ones(self.nc)
        l[self.rows.pg], u[self.rows.pg] = net.pg_min, net.pg_max
        u[self.rows.line_p] = net.fmax
        u[self.rows.line_m] = net.fmax
        l[self.rows.sp] = 0.0
        l[self.rows.sm] = 0.0
        return l, u

    def bounds(self, pd: np.ndarray, params: Params, loss_hat: float):
        l, u = self.base_l.copy(), self.base_u.copy()
        rhs = self.balance_rhs(pd, params.alpha, loss_hat, params.rbias)
        l[self.rows.bal] = rhs
        u[self.rows.bal] = rhs
        u[self.rows.line_p] = self.net.fmax - float(loss_hat) * params.gamma_p
        u[self.rows.line_m] = self.net.fmax - float(loss_hat) * params.gamma_m
        return l, u

    def balance_rhs(self, pd: np.ndarray, alpha: np.ndarray, loss_hat: float,
                    rbias: np.ndarray | None = None) -> np.ndarray:
        rhs = np.asarray(pd, dtype=float).reshape(-1) + np.asarray(alpha).reshape(-1) * float(loss_hat)
        if rbias is not None:
            rhs = rhs + np.asarray(rbias, dtype=float).reshape(-1)
        return rhs

    def flow(self, theta: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(b, dtype=float).reshape(-1) * (self.net.Ar @ np.asarray(theta, dtype=float).reshape(-1))

    def split(self, x: np.ndarray):
        return x[self.v.pg], x[self.v.theta], x[self.v.sp], x[self.v.sm]
