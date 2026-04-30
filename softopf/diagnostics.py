import numpy as np
from .params import Params
from .solution import OPFSolution


def residuals(net, template, pd: np.ndarray, params: Params,
              loss_hat: float, sol: OPFSolution) -> dict:
    flow = template.flow(sol.theta, params.b)
    rhs = template.balance_rhs(pd, params.alpha, loss_hat, params.rbias)
    bal = net.Cg @ sol.pg - net.A.T @ flow - rhs
    gen_low = np.maximum(net.pg_min - sol.pg, 0.0)
    gen_high = np.maximum(sol.pg - net.pg_max, 0.0)
    cap_p = net.fmax - float(loss_hat) * params.gamma_p
    cap_m = net.fmax - float(loss_hat) * params.gamma_m
    line_p = np.maximum(flow - sol.sp - cap_p, 0.0)
    line_m = np.maximum(-flow - sol.sm - cap_m, 0.0)
    return {
        "balance_inf": float(np.linalg.norm(bal, ord=np.inf)),
        "system_balance": float(sol.pg.sum() - pd.sum() - rhs.sum() + pd.sum()),
        "gen_violation": float(max(gen_low.max(), gen_high.max())),
        "line_violation": float(max(line_p.max(), line_m.max())),
        "slack_sum_pos": float(np.maximum(sol.sp, 0.0).sum() + np.maximum(sol.sm, 0.0).sum()),
        "max_loading": float(np.max(np.abs(flow) / np.maximum(net.fmax, 1.0))),
    }
