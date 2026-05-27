"""Compare two models on mice evaluation logs with a paired t-test."""

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


# ============================================================================
# Loading — one scalar per seed, no fold structure
# ============================================================================

def _load_mice_runs(log_dir: Path) -> Dict[str, Dict[Optional[int], float]]:
    """
    Returns: metric -> {seed -> scalar value}
    Warns if a seed produces multiple values and averages them.
    """
    # seed -> metric -> [values]
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

    # Flatten to seed -> single value, warn if multiple
    result: Dict[str, Dict[Optional[int], float]] = defaultdict(dict)

    for seed, metric_dict in buckets.items():
        for metric in METRICS:
            vals = metric_dict.get(metric, [])
            if not vals:
                print(f"[WARNING] seed={seed}: no value for {metric}, skipping.")
                continue
            if len(vals) > 1:
                print(
                    f"[WARNING] seed={seed}: {len(vals)} values for {metric}, "
                    f"averaging."
                )
            result[metric][seed] = float(np.mean(vals))

    return dict(result)


# ============================================================================
# Statistics — reused from compare_models.py
# ============================================================================

def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    std = np.std(diff, ddof=1)
    return 0.0 if std == 0 else float(np.mean(diff) / std)


def _confidence_interval(values: np.ndarray) -> Tuple[float, float]:
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    interval = stats.t.interval(
        0.95, df=n - 1, loc=np.mean(values), scale=stats.sem(values)
    )
    return float(interval[0]), float(interval[1])


def _wilcoxon_test(arr_a: np.ndarray, arr_b: np.ndarray) -> Dict:
    if len(arr_a) < 2:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    result = stats.wilcoxon(
        arr_a, arr_b,
        zero_method="wilcox",
        alternative="two-sided",
        correction=True,
    )
    if not hasattr(result, "statistic") or not hasattr(result, "pvalue"):
        raise ValueError("Unexpected result from wilcoxon test: "
                         f"{result}")
    return {
        "statistic": result.statistic,  # type: ignore
        "pvalue": result.pvalue,        # type: ignore
    }


# ============================================================================
# Comparison
# ============================================================================

def _compare_metric(
    metric: str,
    map_a: Dict[Optional[int], float],
    map_b: Dict[Optional[int], float],
) -> None:

    common_seeds = sorted(
        map_a.keys() & map_b.keys(),
        key=lambda s: (s is None, s),
    )

    if len(common_seeds) < 2:
        print(f"{metric}: fewer than 2 matched seeds, cannot test.")
        return

    arr_a = np.array([map_a[s] for s in common_seeds])
    arr_b = np.array([map_b[s] for s in common_seeds])

    diff = arr_a - arr_b
    t_result = stats.ttest_rel(arr_a, arr_b, nan_policy="omit")
    w_result = _wilcoxon_test(arr_a, arr_b)
    ci_low, ci_high = _confidence_interval(diff)
    d = _cohens_d_paired(arr_a, arr_b)

    print(f"{metric}:")
    print(f"  matched seeds:           {len(common_seeds)}")
    print(f"  model A mean:            {np.mean(arr_a):.6f}  "
          f"(std {np.std(arr_a, ddof=1):.6f})")
    print(f"  model B mean:            {np.mean(arr_b):.6f}  "
          f"(std {np.std(arr_b, ddof=1):.6f})")
    print(f"  mean difference (A - B): {np.mean(diff):.6f}")
    print(f"  paired t-test p-value:   {t_result.pvalue:.6g}")  # type: ignore
    print(f"  Wilcoxon p-value:        {w_result['pvalue']:.6g}")
    print(f"  Wilcoxon statistic:      {w_result['statistic']:.6f}")
    print(f"  effect size (Cohen's d): {d:.6f}")
    print(f"  95% CI of difference:    [{ci_low:.6f}, {ci_high:.6f}]")
    print()


# ============================================================================
# Entry point
# ============================================================================

def main(model_a: str, model_b: str) -> None:
    path_a = Path(model_a)
    path_b = Path(model_b)

    runs_a = _load_mice_runs(path_a)
    runs_b = _load_mice_runs(path_b)

    print("=" * 60)
    print("MICE MODEL COMPARISON")
    print("=" * 60)

    bonferroni_alpha = 0.05 / len(METRICS)
    print(
        f"Bonferroni-corrected significance threshold: "
        f"{bonferroni_alpha:.4f}  (α=0.05 / {len(METRICS)} metrics)\n"
    )

    for metric in METRICS:
        map_a = runs_a.get(metric, {})
        map_b = runs_b.get(metric, {})
        if not map_a or not map_b:
            print(f"{metric}: missing data for one or both models, skipping.")
            continue
        _compare_metric(metric, map_a, map_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paired t-test comparing two models on mice evaluation logs."
    )
    parser.add_argument(
        "model_a",
        nargs="?",
        default=str(HERE / "mice" / "pretrained"),
        help="Path to pretrained mice log directory",
    )
    parser.add_argument(
        "model_b",
        nargs="?",
        default=str(HERE / "mice" / "scratch"),
        help="Path to from-scratch mice log directory",
    )
    args = parser.parse_args()
    main(args.model_a, args.model_b)