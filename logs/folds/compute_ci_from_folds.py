"""Compute confidence intervals for F1 and ROC-AUC from parsed fold summaries.

This script reads the fold lines from the `logs/folds/*.out` files using
the existing `parse_folds.py` logic, extracts `eval_f1` per run (fold+seed),
and computes:
- run-level bootstrap CI (bootstrap over per-run F1 means)
- t-interval over per-run F1s

It saves a histogram of the bootstrap distribution to `ci_f1_run_level.png`.
"""
import os
import importlib.util
from pathlib import Path
from typing import Optional
import numpy as np
import parse_folds

HERE = Path(__file__).parent / "pretrained"


def main(dir: Optional[str] = None, out_dir: Optional[str] = None):
    dir = dir or str(HERE)
    out_dir_path = Path(HERE) if out_dir is None else Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    items = parse_folds.data_from_files(dir)
    folds = []
    for filename, line in items:
        seed = parse_folds.seed_from_filename(filename)
        data = parse_folds.parse_fold(line, seed)
        if data:
            folds.append(data)

    # collect eval_f1 / eval_roc_auc values
    f1s = [float(d['eval_f1']) for d in folds if 'eval_f1' in d]
    roc_aucs = [float(d['eval_roc_auc']) for d in folds if 'eval_roc_auc' in d]
    if not f1s and not roc_aucs:
        print('No eval_f1 or eval_roc_auc values found in parsed folds.')
        return

    # import ci utilities
    from ci_utils import (
        run_level_bootstrap_f1,
        t_interval,
        plot_bootstrap_distribution,
    )

    mean_rb, ci_rb, samples_rb = run_level_bootstrap_f1(f1s, n_boot=5000, random_state=0)
    mean_t, ci_t = t_interval(f1s)

    print('Run-level bootstrap (mean, 95% CI):', mean_rb, ci_rb)
    print('t-interval (mean, 95% CI):', mean_t, ci_t)

    if roc_aucs:
        mean_roc_rb, ci_roc_rb, samples_roc_rb = run_level_bootstrap_f1(roc_aucs, n_boot=5000, random_state=0)
        mean_roc_t, ci_roc_t = t_interval(roc_aucs)
        print('Run-level bootstrap ROC-AUC (mean, 95% CI):', mean_roc_rb, ci_roc_rb)
        print('t-interval ROC-AUC (mean, 95% CI):', mean_roc_t, ci_roc_t)

    # plot and save
    out_path = out_dir_path / 'ci_f1_run_level.png'
    plot_bootstrap_distribution(samples_rb, ci_rb, title='Run-level bootstrap F1', save_path=str(out_path))
    print('Saved bootstrap plot to', out_path)

    if roc_aucs:
        out_path_roc = out_dir_path / 'ci_roc_auc_run_level.png'
        plot_bootstrap_distribution(samples_roc_rb, ci_roc_rb, title='Run-level bootstrap ROC-AUC', save_path=str(out_path_roc))
        print('Saved ROC-AUC bootstrap plot to', out_path_roc)


if __name__ == '__main__':
    main(dir=str(HERE), out_dir=str(HERE))
