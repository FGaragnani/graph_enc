"""Equivalence testing (TOST) for mice evaluation logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

import parse_folds

HERE = Path(__file__).parent
METRICS = ("eval_f1", "eval_roc_auc")

# Margins to sweep if no --delta is provided.
# Chosen to bracket plausible biological relevance thresholds.
DEFAULT_DELTAS = [0.001, 0.002, 0.005, 0.010, 0.020]


# ============================================================================
# Loading (identical to compare_mice.py)
# ============================================================================

def _load_mice_runs(log_dir: Path) -> Dict[str, Dict[Optional[int], float]]:
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

    result: Dict[str, Dict[Optional[int], float]] = defaultdict(dict)
    for seed, metric_dict in buckets.items():
        for metric in METRICS:
            vals = metric_dict.get(metric, [])
            if not vals:
                continue
            if len(vals) > 1:
                print(
                    f"[WARNING] seed={seed}: {len(vals)} values for "
                    f"{metric}, averaging."
                )
            result[metric][seed] = float(np.mean(vals))

    return dict(result)


# ============================================================================
# TOST
# ============================================================================

def _tost(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    delta: float,
    confidence: float = 0.90,   # 90% CI for TOST, not 95%
) -> Dict:
    """
    Two One-Sided Tests for paired samples.

    Runs two one-sided paired t-tests:
        t1: H0: mu_diff <= -delta  (lower bound)
        t2: H0: mu_diff >=  delta  (upper bound)

    Equivalence is concluded when BOTH p-values < alpha (default 0.05),
    which is equivalent to the 90% CI lying within [-delta, +delta].

    Note: TOST uses 90% CI, not 95%, because the two one-sided tests
    each use alpha=0.05. This is not a mistake.
    """
    diff = arr_a - arr_b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    se = float(stats.sem(diff))
    df = n - 1

    # One-sided t-statistics
    t_lower = (mean_diff + delta) / se   # tests H0: mu_diff <= -delta
    t_upper = (mean_diff - delta) / se   # tests H0: mu_diff >= +delta

    # One-sided p-values
    p_lower = float(stats.t.sf(t_lower, df=df))   # P(T > t_lower)
    p_upper = float(stats.t.cdf(t_upper, df=df))  # P(T < t_upper)

    # TOST p-value is the maximum of the two (most conservative)
    p_tost = max(p_lower, p_upper)

    # 90% CI of the difference
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    # Equivalence is concluded when 90% CI ⊆ [-delta, +delta]
    equivalent = ci_low >= -delta and ci_high <= delta

    return {
        "n": n,
        "mean_diff": mean_diff,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "delta": delta,
        "equivalent": equivalent,
    }


# ============================================================================
# Reporting
# ============================================================================

def _print_tost_result(metric: str, r: Dict, alpha: float = 0.05) -> None:
    conclusion = (
        "EQUIVALENT  ✓" if r["equivalent"]
        else "not equivalent at this delta"
    )
    print(f"  delta = {r['delta']:.4f}:")
    print(f"    90% CI of difference: [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}]")
    print(f"    equivalence margin:   [{-r['delta']:+.6f}, {+r['delta']:+.6f}]")
    print(f"    p_lower:  {r['p_lower']:.6g}  "
          f"(H0: diff <= -{r['delta']:.4f})")
    print(f"    p_upper:  {r['p_upper']:.6g}  "
          f"(H0: diff >= +{r['delta']:.4f})")
    print(f"    p_tost:   {r['p_tost']:.6g}  (max of above, threshold {alpha})")
    print(f"    --> {conclusion}")
    print()


def _compare_metric_tost(
    metric: str,
    map_a: Dict[Optional[int], float],
    map_b: Dict[Optional[int], float],
    deltas: List[float],
    alpha: float = 0.05,
) -> None:
    common_seeds = sorted(
        map_a.keys() & map_b.keys(),
        key=lambda s: (s is None, s),
    )

    if len(common_seeds) < 2:
        print(f"{metric}: fewer than 2 matched seeds, cannot test.\n")
        return

    arr_a = np.array([map_a[s] for s in common_seeds])
    arr_b = np.array([map_b[s] for s in common_seeds])

    print(f"{metric}:")
    print(f"  matched seeds:           {len(common_seeds)}")
    print(f"  model A mean:            {np.mean(arr_a):.6f}  "
          f"(std {np.std(arr_a, ddof=1):.6f})")
    print(f"  model B mean:            {np.mean(arr_b):.6f}  "
          f"(std {np.std(arr_b, ddof=1):.6f})")
    print(f"  mean difference (A - B): {np.mean(arr_a - arr_b):+.6f}")
    print()

    for delta in deltas:
        r = _tost(arr_a, arr_b, delta=delta, confidence=0.90)
        _print_tost_result(metric, r, alpha=alpha)


# ============================================================================
# Entry point
# ============================================================================

def main(model_a: str, model_b: str, delta: Optional[float]) -> None:
    path_a = Path(model_a)
    path_b = Path(model_b)

    runs_a = _load_mice_runs(path_a)
    runs_b = _load_mice_runs(path_b)

    deltas = [delta] if delta is not None else DEFAULT_DELTAS

    print("=" * 60)
    print("MICE MODEL COMPARISON — TOST EQUIVALENCE TEST")
    print("=" * 60)
    print(
        "Equivalence is concluded when the 90% CI of the difference\n"
        "lies entirely within [-delta, +delta].\n"
        "Choose delta based on the smallest difference that would\n"
        "matter biologically, NOT based on the observed difference.\n"
    )

    for metric in METRICS:
        map_a = runs_a.get(metric, {})
        map_b = runs_b.get(metric, {})
        if not map_a or not map_b:
            print(f"{metric}: missing data for one or both models, skipping.")
            continue
        _compare_metric_tost(metric, map_a, map_b, deltas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TOST equivalence test for mice evaluation logs."
    )
    parser.add_argument(
        "model_a",
        nargs="?",
        default=str(HERE / "mice" / "pretrained"),
    )
    parser.add_argument(
        "model_b",
        nargs="?",
        default=str(HERE / "mice" / "scratch"),
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help=(
            "Equivalence margin. If omitted, sweeps over "
            f"{DEFAULT_DELTAS}. "
            "Must be chosen from domain knowledge, not from the data."
        ),
    )
    args = parser.parse_args()
    main(args.model_a, args.model_b, args.delta)