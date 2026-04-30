# soft_opf_alpha_step7

Step-7 prototype for the fixed-loss softened DC-OPF training problem.

The forward QP is

```text
Cg Pg - A' diag(b) A_r theta = pd + loss_hat * alpha + rbias
 diag(b) A_r theta - s+ <= fmax - gamma_p * loss_hat
-diag(b) A_r theta - s- <= fmax - gamma_m * loss_hat
s+, s- >= 0
```

The trainable blocks are optimized in this order:

```text
alpha -> rbias -> b -> (gamma_p, gamma_m)
```

Important conventions:

- `alpha` is projected onto the simplex.
- `rbias` is initialized at zero and is **not** constrained to have zero sum.
- `b` is box-constrained by `--b-min-frac` and `--b-max-frac` relative to the physical susceptance.
- `gamma_p` and `gamma_m` are box-constrained in `[0, --gamma-max]`.
- The line search is checked by true full-training-set forward QP solves, not by a reduced-KKT prediction.

## Run

Place the MATLAB dataset in `data/`, then run:

```bash
python scripts/07_train_step7.py --mat data/case39.mat --backend cvxpy --solver GUROBI
```

The learned parameters and training log are saved in:

```text
outputs/step7/
```

Evaluate a saved parameter file with:

```bash
python scripts/08_evaluate_step7.py \
  --mat data/case39.mat \
  --params outputs/step7/params_step7.npz \
  --backend cvxpy --solver GUROBI
```

For a single-sample overfitting diagnostic, use `--n-train 1 --n-val 0 --batch-size 0`.
