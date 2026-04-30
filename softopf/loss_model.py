from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class FlowQuadraticLossModel:
    """Offline estimator: L_hat = beta0 + sum_l beta_l f0_l^2."""
    beta0: float
    beta: np.ndarray
    rmse: float
    mae: float


def fit_flow_loss_model(f0: np.ndarray, lac: np.ndarray,
                        ridge: float = 1e-6,
                        nonnegative: bool = True) -> FlowQuadraticLossModel:
    """Fit the fixed-loss estimator; returned coefficients are in MW units."""
    f0 = np.asarray(f0, dtype=float)
    lac = np.asarray(lac, dtype=float).reshape(-1)
    Xraw = f0 * f0
    scale = np.maximum(np.sqrt(np.mean(Xraw * Xraw, axis=0)), 1.0)
    X = Xraw / scale

    A = np.c_[np.ones(X.shape[0]), X]
    y = lac
    if ridge > 0.0:
        A = np.vstack([A, np.c_[np.zeros(X.shape[1]), np.sqrt(ridge) * np.eye(X.shape[1])]])
        y = np.r_[y, np.zeros(X.shape[1])]

    coef = lsq_linear(A, y, bounds=(0.0, np.inf) if nonnegative else (-np.inf, np.inf),
                      lsmr_tol="auto").x
    beta0 = float(coef[0])
    beta = coef[1:] / scale
    pred = beta0 + Xraw @ beta
    return FlowQuadraticLossModel(
        beta0=beta0,
        beta=beta,
        rmse=float(np.sqrt(np.mean((pred - lac) ** 2))),
        mae=float(np.mean(np.abs(pred - lac))),
    )


def loss_values(model: FlowQuadraticLossModel, f0: np.ndarray) -> np.ndarray:
    f0 = np.asarray(f0, dtype=float)
    return model.beta0 + (f0 * f0) @ model.beta
