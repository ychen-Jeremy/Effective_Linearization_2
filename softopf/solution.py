from dataclasses import dataclass
import numpy as np


@dataclass
class OPFSolution:
    x: np.ndarray
    pg: np.ndarray
    theta: np.ndarray
    sp: np.ndarray
    sm: np.ndarray
    obj: float
    status: str
    iter: int = 0
    y: np.ndarray | None = None

    @property
    def slack_sum(self) -> float:
        return float(self.sp.sum() + self.sm.sum())
