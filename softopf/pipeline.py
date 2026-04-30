from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .config import Settings
from .data import Dataset, load_dataset
from .network import Network, build_network
from .baseline import baseline_flows
from .loss_model import FlowQuadraticLossModel, fit_flow_loss_model, loss_values
from .params import Params, init_params
from .template import SoftQPTemplate


@dataclass
class Problem:
    ds: Dataset
    net: Network
    f0: np.ndarray
    loss_model: FlowQuadraticLossModel
    loss_hat: np.ndarray
    params: Params
    template: SoftQPTemplate


def build_problem(mat_path: str | Path,
                  case_path: str | Path | None = None,
                  settings: Settings | None = None,
                  alpha_mode: str = "capacity") -> Problem:
    settings = settings or Settings()
    ds = load_dataset(mat_path, case_path)
    net = build_network(ds.mpc0, ds.meta)
    f0 = baseline_flows(ds, net)
    loss_model = fit_flow_loss_model(f0, ds.lac, ridge=settings.loss_ridge)
    loss_hat = loss_values(loss_model, f0)
    params = init_params(net, alpha_mode)
    template = SoftQPTemplate(net, settings)
    return Problem(ds=ds, net=net, f0=f0, loss_model=loss_model,
                   loss_hat=loss_hat, params=params, template=template)


def make_solver(problem: Problem, backend: str = "cvxpy"):
    backend = backend.lower()
    if backend == "cvxpy":
        from .solver_cvxpy import CvxpyForwardSolver
        st = problem.template.settings
        return CvxpyForwardSolver(problem.template, solver=st.cvxpy_solver,
                                  verbose=st.cvxpy_verbose)
    if backend == "osqp":
        from .solver_osqp import SoftOPFSolver
        return SoftOPFSolver(problem.template)
    raise ValueError("backend must be 'cvxpy' or 'osqp'")


def solve_one(problem: Problem, solver, sample_id: int, backend: str):
    pd = problem.ds.pd[sample_id]
    lh = problem.loss_hat[sample_id]
    return solver.solve_one(pd, problem.params, lh, sample_id=sample_id)


def solve_indices(problem: Problem, solver, indices, backend: str):
    idx = np.asarray(indices, dtype=int)
    return solver.solve_batch(problem.ds.pd[idx], problem.params, problem.loss_hat[idx], sample_ids=idx)
