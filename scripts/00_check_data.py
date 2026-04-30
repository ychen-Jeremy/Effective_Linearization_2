from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def resolve(path):
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


from softopf.config import Settings
from softopf.pipeline import build_problem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default="data/case39.mat")
    p.add_argument("--case", default=None)
    p.add_argument("--alpha-mode", default="capacity", choices=["capacity", "uniform"])
    args = p.parse_args()

    prob = build_problem(resolve(args.mat), resolve(args.case), Settings(), args.alpha_mode)
    beta = prob.loss_model.beta
    print("[Data / network summary]")
    print(f"  samples                 : {len(prob.ds)}")
    print(f"  buses / branches / gens : {prob.net.n} / {prob.net.m} / {prob.net.g}")
    print(f"  alpha sum/min/max       : {prob.params.alpha.sum():.8g} / {prob.params.alpha.min():.3e} / {prob.params.alpha.max():.3e}")
    print(f"  beta0                  : {prob.loss_model.beta0:.6g}")
    print(f"  beta nnz/min/max       : {(beta > 1e-12).sum()} / {beta.min():.3e} / {beta.max():.3e}")
    print(f"  loss RMSE / MAE        : {prob.loss_model.rmse:.6g} / {prob.loss_model.mae:.6g} MW")
    print(f"  first true/hat loss    : {prob.ds.lac[0]:.6g} / {prob.loss_hat[0]:.6g} MW")
    print(f"  QP variables / rows    : {prob.template.nx} / {prob.template.nc}")


if __name__ == "__main__":
    main()
