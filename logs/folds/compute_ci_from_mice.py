"""Compute confidence intervals for F1 and ROC-AUC from mice evaluation logs.

Each log file corresponds to one model trained with one seed, evaluated
on the mice test set. Since there is no cross-validation here, each log
produces exactly one scalar F1 and one scalar ROC-AUC. The CI is then
estimated across seeds.

Usage:
    python ci_mice.py                        # reads from ./mice/
    python ci_mice.py --logs path/to/logs    # custom log directory
    python ci_mice.py --logs path/to/logs --out path/to/output
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

import parse_folds

HERE = Path(__file__).parent
METRICS = ("eval_f1", "eval_roc_auc")
DEFAULT_DIR = HERE / "mice" / "pretrained"

# ============================================================================
# Parsing
# ============================================================================

def _load_mice_runs(log_dir: Path) -> Dict[str, List[float]]:
    """
    Parse all .out files in log_dir.
    Returns a dict: metric_name -> list of scalar values, one per seed.

    Warns if a seed produces more than one value for a metric (unexpected
    fold structure) and averages them, or if a seed produces no value.
    """
    # bucket: seed -> metric -> [values]
    buckets: Dict[Optional[int], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    files = sorted(log_dir.glob("*.out"))

    if not files:
        raise FileNotFoundError(f"No .out files found in {log_dir}")

    for file_path in files:
        items = parse_folds.data_from_files(str(log_dir))
        items = [
            (fname, line) for fname, line in items
            if fname == file_path.name
        ]

        for filename, line in items:
            seed = parse_folds.seed_from_filename(filename)
            data = parse_folds.parse_fold(line, seed)

            if not data:
                continue

            for metric in METRICS:
                if metric in data and data[metric] is not None:
                    buckets[seed][metric].append(float(data[metric]))

    # Flatten: one value per seed per metric
    metric_values: Dict[str, List[float]] = defaultdict(list)

    for seed, metric_dict in buckets.items():
        for metric in METRICS:
            vals = metric_dict.get(metric, [])

            if not vals:
                print(f"[WARNING] seed={seed}: no value for {metric}, skipping.")
                continue

            if len(vals) > 1:
                print(
                    f"[WARNING] seed={seed}: {len(vals)} values for {metric} "
                    f"(expected 1). Averaging them."
                )

            metric_values[metric].append(float(np.mean(vals)))

    return dict(metric_values)


# ============================================================================
# Statistics
# ============================================================================

def _t_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high)."""
    n = len(values)
    mean = float(np.mean(values))

    if n < 2:
        return mean, float("nan"), float("nan")

    se = float(stats.sem(values))
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)

    return mean, float(mean - t_crit * se), float(mean + t_crit * se)


def _run_level_bootstrap(
    values: np.ndarray,
    n_boot: int = 5000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> Tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high)."""
    rng = np.random.default_rng(random_state)
    n = len(values)
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


# ============================================================================
# Reporting
# ============================================================================

def _print_metric_report(
    metric: str,
    values: np.ndarray,
) -> None:
    n = len(values)

    t_mean, t_lo, t_hi = _t_interval(values)
    b_mean, b_lo, b_hi = _run_level_bootstrap(values)

    std = float(np.std(values, ddof=1)) if n > 1 else float("nan")

    print(f"  {metric}:")
    print(f"    n seeds:          {n}")
    print(f"    mean:             {t_mean:.6f}")
    print(f"    std:              {std:.6f}")
    print(f"    t-interval 95%:   [{t_lo:.6f}, {t_hi:.6f}]")

    # Remind the user bootstrap is unreliable at small n
    if n < 10:
        print(
            f"    bootstrap 95%:    [{b_lo:.6f}, {b_hi:.6f}]"
            f"  (prefer t-interval at n={n})"
        )
    else:
        print(f"    bootstrap 95%:    [{b_lo:.6f}, {b_hi:.6f}]")

    print()


# ============================================================================
# Entry point
# ============================================================================

def main(log_dir: str, out_dir: Optional[str] = None) -> None:
    log_path = Path(log_dir)

    print("=" * 60)
    print("MICE EVALUATION — CONFIDENCE INTERVALS")
    print("=" * 60)
    print(f"Log directory: {log_path}")
    print()

    metric_values = _load_mice_runs(log_path)

    if not metric_values:
        print("No metric values found. Check log format.")
        return

    for metric in METRICS:
        if metric not in metric_values:
            print(f"  {metric}: not found in any log file, skipping.")
            continue

        values = np.array(metric_values[metric])
        _print_metric_report(metric, values)

    # Optionally save raw values for further analysis
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for metric, values in metric_values.items():
            out_file = out_path / f"mice_{metric}_seed_values.txt"
            np.savetxt(out_file, values, header=metric, comments="# ")
            print(f"Saved seed-level values to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CI for mice evaluation logs (one log per seed)."
    )
    parser.add_argument(
        "--logs",
        default=str(DEFAULT_DIR),
        help="Directory containing .out log files (default: ./mice/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional directory to save seed-level values as .txt files.",
    )
    args = parser.parse_args()
    main(args.logs, args.out)