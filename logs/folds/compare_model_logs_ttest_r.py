"""Equivalence testing (TOST) for model log sets (fold-aware)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import stats

import parse_folds


HERE = Path(__file__).parent
METRICS = ("eval_f1", "eval_roc_auc")

# Margins to sweep if no --delta is provided.
# Chosen to bracket plausible relevance thresholds.
DEFAULT_DELTAS = [0.001, 0.002, 0.005, 0.010, 0.020]


# ============================================================================
# Data structures (identical to compare_model_logs_ttest.py)
# ============================================================================

@dataclass(frozen=True)
class RunKey:
    seed: Optional[int]
    fold: int


@dataclass
class ModelRun:
    key: RunKey
    metrics: Dict[str, float]
    source: str


# ============================================================================
# Loading (identical to compare_model_logs_ttest.py)
# ============================================================================

def _iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(path.glob("*.out"))
    else:
        yield path


def _load_model_runs(path: Path) -> List[ModelRun]:
    runs: List[ModelRun] = []
    seen_keys = set()

    for file_path in _iter_input_files(path):

        if not file_path.exists():
            continue

        items = parse_folds.data_from_files(
            str(file_path.parent if file_path.is_file() else file_path)
        )

        if file_path.is_file():
            items = [
                (filename, line)
                for filename, line in items
                if filename == file_path.name
            ]

        for filename, line in items:

            seed = parse_folds.seed_from_filename(filename)
            data = parse_folds.parse_fold(line, seed)

            if not data:
                continue

            fold = int(data.get("fold"))  # type: ignore

            metrics = {
                name: float(data[name])
                for name in METRICS
                if name in data and data[name] is not None
            }

            if not metrics:
                continue

            key = RunKey(seed=seed, fold=fold)

            if key in seen_keys:
                print(
                    f"[WARNING] Duplicate run detected: "
                    f"(seed={seed}, fold={fold}) in {filename}"
                )

            seen_keys.add(key)

            runs.append(
                ModelRun(
                    key=key,
                    metrics=metrics,
                    source=filename,
                )
            )

    return runs


def _build_metric_map(
    runs: List[ModelRun],
    metric: str,
) -> Dict[RunKey, float]:
    return {
        run.key: run.metrics[metric]
        for run in runs
        if metric in run.metrics
    }


def _aggregate_by_seed(
    metric_map: Dict[RunKey, float],
) -> Dict[Optional[int], float]:
    """Average per-fold values within each seed to get independent observations."""
    from collections import defaultdict
    buckets: Dict[Optional[int], List[float]] = defaultdict(list)
    for key, value in metric_map.items():
        buckets[key.seed].append(value)
    return {seed: float(np.mean(vals)) for seed, vals in buckets.items()}


# ============================================================================
# TOST
# ============================================================================

def _tost_paired(
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

    t_lower = (mean_diff + delta) / se   # tests H0: mu_diff <= -delta
    t_upper = (mean_diff - delta) / se   # tests H0: mu_diff >= +delta

    p_lower = float(stats.t.sf(t_lower, df=df))   # P(T > t_lower)
    p_upper = float(stats.t.cdf(t_upper, df=df))  # P(T < t_upper)

    p_tost = max(p_lower, p_upper)

    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    equivalent = ci_low >= -delta and ci_high <= delta

    return {
        "test": "tost_paired",
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


def _tost_welch(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    delta: float,
    confidence: float = 0.90,
) -> Dict:
    """
    Two One-Sided Tests for independent samples (Welch fallback).

    Uses Welch-Satterthwaite degrees of freedom.
    """
    n_a = len(arr_a)
    n_b = len(arr_b)

    mean_diff = float(np.mean(arr_a) - np.mean(arr_b))

    var_a = np.var(arr_a, ddof=1)
    var_b = np.var(arr_b, ddof=1)

    se = float(np.sqrt(var_a / n_a + var_b / n_b))

    # Welch-Satterthwaite degrees of freedom
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (
        ((var_a / n_a) ** 2) / (n_a - 1)
        + ((var_b / n_b) ** 2) / (n_b - 1)
    )
    df = df_num / df_den

    t_lower = (mean_diff + delta) / se
    t_upper = (mean_diff - delta) / se

    p_lower = float(stats.t.sf(t_lower, df=df))
    p_upper = float(stats.t.cdf(t_upper, df=df))

    p_tost = max(p_lower, p_upper)

    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=df)
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    equivalent = ci_low >= -delta and ci_high <= delta

    return {
        "test": "tost_welch",
        "n_a": n_a,
        "n_b": n_b,
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

def _print_tost_result(r: Dict, alpha: float = 0.05) -> None:
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
    model_a: List[ModelRun],
    model_b: List[ModelRun],
    metric: str,
    deltas: List[float],
    alpha: float = 0.05,
) -> None:

    metric_a = _build_metric_map(model_a, metric)
    metric_b = _build_metric_map(model_b, metric)

    if not metric_a or not metric_b:
        print(f"{metric}: missing values, skipping\n")
        return

    agg_a = _aggregate_by_seed(metric_a)
    agg_b = _aggregate_by_seed(metric_b)

    common_seeds = sorted(
        agg_a.keys() & agg_b.keys(),
        key=lambda s: (s is None, s),
    )

    # ------------------------------------------------------------------
    # Paired TOST on per-seed means
    # ------------------------------------------------------------------

    if len(common_seeds) >= 2:

        arr_a = np.asarray([agg_a[s] for s in common_seeds], dtype=float)
        arr_b = np.asarray([agg_b[s] for s in common_seeds], dtype=float)

        print(f"{metric}:")
        print(f"  test:                    tost_paired")
        print(f"  matched seeds:           {len(common_seeds)}")
        print(f"  model A mean:            {np.mean(arr_a):.6f}  "
              f"(std {np.std(arr_a, ddof=1):.6f})")
        print(f"  model B mean:            {np.mean(arr_b):.6f}  "
              f"(std {np.std(arr_b, ddof=1):.6f})")
        print(f"  mean difference (A - B): {np.mean(arr_a - arr_b):+.6f}")
        print()

        for delta in deltas:
            r = _tost_paired(arr_a, arr_b, delta=delta)
            _print_tost_result(r, alpha=alpha)

        return

    # ------------------------------------------------------------------
    # Welch TOST fallback for unmatched seeds
    # ------------------------------------------------------------------

    arr_a = np.asarray(list(agg_a.values()), dtype=float)
    arr_b = np.asarray(list(agg_b.values()), dtype=float)

    if len(arr_a) < 2 or len(arr_b) < 2:
        print(f"{metric}: insufficient data\n")
        return

    print(f"{metric}:")
    print(f"  test:                    tost_welch (Welch fallback — unmatched seeds)")
    print(f"  model A runs:            {len(arr_a)}")
    print(f"  model B runs:            {len(arr_b)}")
    print(f"  model A mean:            {np.mean(arr_a):.6f}  "
          f"(std {np.std(arr_a, ddof=1):.6f})")
    print(f"  model B mean:            {np.mean(arr_b):.6f}  "
          f"(std {np.std(arr_b, ddof=1):.6f})")
    print(f"  mean difference (A - B): {np.mean(arr_a - arr_b):+.6f}")
    print()

    for delta in deltas:
        r = _tost_welch(arr_a, arr_b, delta=delta)
        _print_tost_result(r, alpha=alpha)


# ============================================================================
# Entry point
# ============================================================================

def main(model_a: str, model_b: str, delta: Optional[float]) -> None:

    path_a = Path(model_a)
    path_b = Path(model_b)

    runs_a = _load_model_runs(path_a)
    runs_b = _load_model_runs(path_b)

    if not runs_a:
        print(f"No fold summaries found for model A at {path_a}")
        return

    if not runs_b:
        print(f"No fold summaries found for model B at {path_b}")
        return

    deltas = [delta] if delta is not None else DEFAULT_DELTAS

    print("=" * 60)
    print("MODEL COMPARISON — TOST EQUIVALENCE TEST")
    print("=" * 60)
    print(
        "Equivalence is concluded when the 90% CI of the difference\n"
        "lies entirely within [-delta, +delta].\n"
        "Choose delta based on the smallest difference that would\n"
        "matter practically, NOT based on the observed difference.\n"
    )
    print(f"Model A runs: {len(runs_a)}")
    print(f"Model B runs: {len(runs_b)}")
    print()

    for metric in METRICS:
        _compare_metric_tost(runs_a, runs_b, metric, deltas)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TOST equivalence test for model log sets (fold-aware)."
    )

    parser.add_argument(
        "model_a",
        nargs="?",
        default=str(HERE / "pretrained"),
        help="Path to model A log file or directory",
    )

    parser.add_argument(
        "model_b",
        nargs="?",
        default=str(HERE / "scratch"),
        help="Path to model B log file or directory",
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
