import numpy as np
from .data import Dataset
from .network import Network


def dc_flows_from_dispatch(net: Network, pd: np.ndarray, pg: np.ndarray) -> np.ndarray:
    """Recover DC branch flows from fixed dispatch and load."""
    pd = np.atleast_2d(pd)
    pg = np.atleast_2d(pg)
    inj = (net.Cg @ pg.T).T - pd
    theta = np.linalg.solve(net.Bbus[net.nonref, :].toarray(), inj[:, net.nonref].T).T
    return (net.Bf @ theta.T).T


def baseline_flows(ds: Dataset, net: Network) -> np.ndarray:
    """First-pass branch flows for fixed-loss estimation."""
    if ds.pf_dc.shape[1] == net.m:
        return ds.pf_dc
    return dc_flows_from_dispatch(net, ds.pd, ds.pg_dc)
