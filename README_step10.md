# Step 10: per-sample hybrid line search for RHS blocks

This version is based on the Step-7 block-coordinate training code, with one targeted acceleration:

- `alpha`, `rbias`, and `gamma` are RHS-only blocks. For these blocks the QP matrix is unchanged. The code builds a fixed-active-set affine solution path for each training sample and computes a certified no-switch radius for each sample separately.
- During Armijo line search, a sample inside its own safe region is evaluated by the exact affine/quadratic model and is not re-solved. A sample outside its own safe region is re-solved by the true forward QP at the trial point.
- `b` changes the branch-flow matrix, so it keeps the original true forward-QP Armijo line search.

Main entry point:

```bash
python scripts/10_train_step10_hybrid.py --mat data/case39.mat --case data/case39.csv --full-batch --backend cvxpy
```

Important tuning knobs:

- `--region-safety`: multiplies each sample's certified critical-region radius before declaring that sample safe. Smaller values are more conservative.
- `--critical-margin`, `--critical-dual-tol`: primal and dual margins used in the no-switch certificate.
- `--lr-alpha`, `--lr-rbias`, `--lr-b`, `--lr-gamma`: block initial step scales.
- `--max-trials`, `--backtrack`, `--armijo-c`: Armijo line-search controls.

The training log reports `mode=quad` when all RHS samples are safe, `mode=hybrid` when only unsafe RHS samples are re-solved, and `mode=qp` for the `b` block. For RHS blocks, `qp` in the log is the number of unsafe samples re-solved in that accepted/rejected trial.
