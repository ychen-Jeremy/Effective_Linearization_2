import numpy as np
from .solution import OPFSolution


def dispatch_mse(pg_pred: np.ndarray, pg_ac: np.ndarray) -> float:
    diff = pg_pred - pg_ac
    return float(np.mean(np.sum(diff * diff, axis=1)))


def objective_parts(template, sol: OPFSolution) -> dict:
    net, st = template.net, template.settings
    gen_cost = float(np.sum(net.c2 * sol.pg**2 + net.c1 * sol.pg))
    slack_l1 = float(st.rho_s * np.sum(sol.sp + sol.sm))
    slack_l2 = float(0.5 * st.eps_s * (sol.sp @ sol.sp + sol.sm @ sol.sm))
    return {"gen_cost": gen_cost, "slack_l1": slack_l1,
            "slack_l2": slack_l2, "total": gen_cost + slack_l1 + slack_l2}


def batch_metrics(solutions: list[OPFSolution], pg_ac: np.ndarray) -> dict:
    pg = np.vstack([s.pg for s in solutions])
    return {
        "dispatch_mse": dispatch_mse(pg, pg_ac),
        "mean_slack": float(np.mean([s.slack_sum for s in solutions])),
        "statuses": [s.status for s in solutions],
    }
