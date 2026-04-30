from dataclasses import dataclass
import numpy as np
from .network import Network


@dataclass
class Params:
    # Step 7 parameters in the fixed-loss softened OPF:
    #   Cg Pg - A' diag(b) A_r theta = pd + loss_hat * alpha + rbias
    #   +diag(b) A_r theta - s+ <= fmax - gamma_p * loss_hat
    #   -diag(b) A_r theta - s- <= fmax - gamma_m * loss_hat
    alpha: np.ndarray
    rbias: np.ndarray
    b: np.ndarray
    gamma_p: np.ndarray
    gamma_m: np.ndarray


def project_alpha(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {alpha >= 0, 1^T alpha = 1}."""
    v = np.asarray(v, dtype=float).reshape(-1)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / (np.arange(len(v)) + 1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def init_alpha(net: Network, mode: str = "capacity") -> np.ndarray:
    if mode == "uniform":
        return np.ones(net.n) / net.n
    cap = np.asarray(net.Cg @ net.pg_max).reshape(-1)
    return project_alpha(cap / cap.sum())


def init_params(net: Network, alpha_mode: str = "capacity") -> Params:
    return Params(
        alpha=init_alpha(net, alpha_mode),
        rbias=np.zeros(net.n),       # deliberately unconstrained in Step 7
        b=net.bphys.copy(),
        gamma_p=np.zeros(net.m),
        gamma_m=np.zeros(net.m),
    )
