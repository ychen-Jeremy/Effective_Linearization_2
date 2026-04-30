from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Dataset:
    """Sample-major OPF dataset."""

    pd: np.ndarray        # N x n, MW
    qd: np.ndarray        # N x n, MVAr
    pg_ac: np.ndarray     # N x g, MW
    pg_dc: np.ndarray     # N x g, MW
    pf_ac: np.ndarray     # N x m, MW; may be empty for CSV-only data
    pf_dc: np.ndarray     # N x m, MW; may be empty for CSV-only data
    lac: np.ndarray       # N, sum(PgAC)-sum(Pd), MW
    ldc: np.ndarray       # N, sum(PgDC)-sum(Pd), MW
    obj_ac: np.ndarray    # N
    obj_dc: np.ndarray    # N
    meta: dict
    mpc0: dict

    def __len__(self) -> int:
        return self.pd.shape[0]

    
    def Q(self) -> int:
        """Number of samples; kept for compatibility with training scripts."""
        return self.pd.shape[0]


def _mat(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _vec(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _load_mat_scipy(path: Path) -> dict:
    import scipy.io as sio
    return {k: v for k, v in sio.loadmat(path, squeeze_me=True,
                                         struct_as_record=False,
                                         simplify_cells=True).items()
            if not k.startswith("__")}


def _load_mat_h5(path: Path) -> dict:
    """Read this project's numeric MATLAB v7.3 structs."""
    import h5py

    def read(obj):
        import h5py as _h5py
        if isinstance(obj, _h5py.Dataset):
            arr = np.array(obj)
            if arr.dtype.kind in "fiu":
                arr = arr.T
                return arr.squeeze() if arr.size == 1 else arr
            return arr
        return {k: read(obj[k]) for k in obj.keys() if k != "#refs#"}

    with h5py.File(path, "r") as f:
        return {k: read(f[k]) for k in f.keys() if k != "#refs#"}


def load_mat_any(path: str | Path) -> dict:
    path = Path(path)
    try:
        return _load_mat_scipy(path)
    except NotImplementedError:
        return _load_mat_h5(path)


def _load_mat_dataset(path: Path) -> Dataset:
    obj = load_mat_any(path)
    d, meta, mpc0 = obj["data"], obj["meta"], obj["mpc0"]
    n_sample = len(_vec(d["LAC_MW"]))
    return Dataset(
        pd=_mat(d["Pd_MW"]),
        qd=_mat(d["Qd_MVAr"]),
        pg_ac=_mat(d["PgAC_MW"]),
        pg_dc=_mat(d["PgDC_MW"]),
        pf_ac=_mat(d.get("PfAC_MW", np.zeros((n_sample, 0)))),
        pf_dc=_mat(d.get("PfDC_MW", np.zeros((n_sample, 0)))),
        lac=_vec(d["LAC_MW"]),
        ldc=_vec(d["LDC_MW"]),
        obj_ac=_vec(d["objAC"]),
        obj_dc=_vec(d["objDC"]),
        meta=meta,
        mpc0=mpc0,
    )


def _cols(df, prefix: str):
    return [c for c in df.columns if c.startswith(prefix)]


def _load_csv_dataset(path: Path, case_path: str | Path) -> Dataset:
    import pandas as pd
    df = pd.read_csv(path)
    case = load_mat_any(case_path)
    meta = case.get("meta", {})
    mpc0 = case.get("mpc0", case.get("mpc", case))

    pd_cols = _cols(df, "Pd_bus")
    qd_cols = _cols(df, "Qd_bus")
    pgac_cols = _cols(df, "PgAC_gen")
    pgdc_cols = _cols(df, "PgDC_gen")
    pfac_cols = _cols(df, "PfAC_branch") + _cols(df, "PfAC_line")
    pfdc_cols = _cols(df, "PfDC_branch") + _cols(df, "PfDC_line")

    n = len(df)
    return Dataset(
        pd=df[pd_cols].to_numpy(float),
        qd=df[qd_cols].to_numpy(float) if qd_cols else np.zeros((n, len(pd_cols))),
        pg_ac=df[pgac_cols].to_numpy(float),
        pg_dc=df[pgdc_cols].to_numpy(float),
        pf_ac=df[pfac_cols].to_numpy(float) if pfac_cols else np.zeros((n, 0)),
        pf_dc=df[pfdc_cols].to_numpy(float) if pfdc_cols else np.zeros((n, 0)),
        lac=df["LAC_MW"].to_numpy(float),
        ldc=df["LDC_MW"].to_numpy(float),
        obj_ac=df["objAC"].to_numpy(float),
        obj_dc=df["objDC"].to_numpy(float),
        meta=meta,
        mpc0=mpc0,
    )


def load_dataset(path: str | Path, case_path: str | Path | None = None) -> Dataset:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        if case_path is None:
            raise ValueError("CSV loading needs a companion case .mat path.")
        return _load_csv_dataset(path, case_path)
    return _load_mat_dataset(path)


def batch_indices(n: int, batch_size: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    while True:
        yield rng.choice(n, size=batch_size, replace=False)
