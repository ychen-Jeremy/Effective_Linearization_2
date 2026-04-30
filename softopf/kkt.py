from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .active_set import ActiveSet, active_kkt_equalities
from .params import Params
from .template import SoftQPTemplate


@dataclass
class ActiveKKTSystem:
    """Factorized equality-KKT system for one fixed active set.

    For fixed active constraints and fixed b,
        [P  G_A'] [x ] = [-q]
        [G_A  0 ] [mu]   [ h].
    """
    template: SoftQPTemplate
    active: ActiveSet
    G: sp.csr_matrix
    h: np.ndarray
    labels: list[str]
    K: sp.csc_matrix
    lu: spla.SuperLU

    @property
    def nx(self) -> int:
        return self.template.nx

    @property
    def na(self) -> int:
        return self.G.shape[0]

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(rhs, dtype=float))

    def solve_kkt(self, h: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        rhs = np.r_[-self.template.q, self.h if h is None else h]
        z = self.solve(rhs)
        return z[:self.nx], z[self.nx:]

    def sensitivity(self, dh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.solve(np.r_[np.zeros(self.nx), dh])
        return z[:self.nx], z[self.nx:]

    def adjoint(self, grad_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rhs = np.r_[np.asarray(grad_x, dtype=float), np.zeros(self.na)]
        w = self.lu.solve(rhs, trans="T")
        return w[:self.nx], w[self.nx:]


def build_active_kkt_system(template: SoftQPTemplate, active: ActiveSet,
                            pd: np.ndarray, params: Params, loss_hat: float) -> ActiveKKTSystem:
    G, h, labels = active_kkt_equalities(template, active, pd, params, loss_hat)
    K = sp.bmat([[template.P, G.T], [G, None]], format="csc")
    return ActiveKKTSystem(template=template, active=active, G=G, h=h,
                           labels=labels, K=K, lu=spla.splu(K))


def kkt_residual_norm(system: ActiveKKTSystem, x: np.ndarray, mu: np.ndarray,
                      h: np.ndarray | None = None) -> dict:
    h = system.h if h is None else h
    stat = system.template.P @ x + system.template.q + system.G.T @ mu
    prim = system.G @ x - h
    return {
        "stationarity_inf": float(np.linalg.norm(stat, ord=np.inf)),
        "active_primal_inf": float(np.linalg.norm(prim, ord=np.inf)),
    }


def active_rhs_delta_for_alpha(template: SoftQPTemplate, d_alpha: np.ndarray,
                               loss_hat: float, n_active_rows: int) -> np.ndarray:
    dh = np.zeros(n_active_rows)
    dh[:template.net.n] = float(loss_hat) * np.asarray(d_alpha, dtype=float).reshape(-1)
    return dh


def active_rhs_delta_for_rbias(template: SoftQPTemplate, d_rbias: np.ndarray,
                               n_active_rows: int) -> np.ndarray:
    dh = np.zeros(n_active_rows)
    dh[:template.net.n] = np.asarray(d_rbias, dtype=float).reshape(-1)
    return dh
