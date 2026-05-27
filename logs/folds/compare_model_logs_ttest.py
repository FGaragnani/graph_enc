"""Compare two model log sets with statistically safer tests on per-run metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from scipy import stats
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy import stats

import parse_folds


HERE = Path(__file__).parent
METRICS = ("eval_f1", "eval_roc_auc")


# ============================================================================
# Data structures
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
# Utilities
# ============================================================================

def _iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(path.glob("*.out"))
    else:
        yield path

def _wilcoxon_test(arr_a: np.ndarray, arr_b: np.ndarray) -> Dict:
    """
    Wilcoxon signed-rank test for paired samples.
    More robust than paired t-test for non-normal distributions.
    """

    # scipy requires no NaNs and equal length
    mask = ~np.isnan(arr_a) & ~np.isnan(arr_b)
    arr_a = arr_a[mask]
    arr_b = arr_b[mask]

    if len(arr_a) < 2:
        return {
            "test": "wilcoxon",
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "n": len(arr_a),
        }

    result = stats.wilcoxon(
        arr_a,
        arr_b,
        zero_method="wilcox",
        alternative="two-sided",
        correction=True,
    )

    diff = arr_a - arr_b

    return {
        "test": "wilcoxon",
        "statistic": float(result.statistic),   # type: ignore
        "pvalue": float(result.pvalue),         # type: ignore
        "n": len(arr_a),
        "mean_diff": float(np.mean(diff)),
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

def _mann_whitney_test(arr_a: np.ndarray, arr_b: np.ndarray) -> Dict:
    """Mann-Whitney U for unpaired/unequal-length samples."""
    result = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
    return {
        "test": "mann_whitney",
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n_a": len(arr_a),
        "n_b": len(arr_b),
        "mean_diff": float(np.mean(arr_a) - np.mean(arr_b)),
    }


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
            
            fold = int(data.get("fold")) # type: ignore

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

    metric_map: Dict[RunKey, float] = {}

    for run in runs:
        if metric in run.metrics:
            metric_map[run.key] = run.metrics[metric]

    return metric_map


# ============================================================================
# Statistics
# ============================================================================

def _cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for independent samples."""

    n1 = len(a)
    n2 = len(b)

    if n1 < 2 or n2 < 2:
        return float("nan")

    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)

    pooled_std = np.sqrt(
        ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    )

    if pooled_std == 0:
        return 0.0

    return (np.mean(a) - np.mean(b)) / pooled_std


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Cohen's dz."""

    diff = a - b

    std = np.std(diff, ddof=1)

    if std == 0:
        return 0.0

    return float(np.mean(diff) / std)


def _confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:

    n = len(values)

    if n < 2:
        return float("nan"), float("nan")

    mean = np.mean(values)
    sem = stats.sem(values)

    interval = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean,
        scale=sem,
    )

    return float(interval[0]), float(interval[1])


def _bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> Tuple[float, float]:

    rng = np.random.default_rng(rng_seed)

    means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    alpha = 1.0 - confidence

    low = np.percentile(means, 100 * alpha / 2)
    high = np.percentile(means, 100 * (1 - alpha / 2))

    return float(low), float(high)


def _paired_test(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
) -> Dict:

    t_result = stats.ttest_rel(arr_a, arr_b, nan_policy="omit")

    diff = arr_a - arr_b
    ci_low, ci_high = _confidence_interval(diff)
    boot_low, boot_high = _bootstrap_ci(diff)

    return {
        "test": "paired",
        "n": len(arr_a),
        "statistic": float(t_result.statistic), # type: ignore
        "pvalue": float(t_result.pvalue),       # type: ignore
        "effect_size": float(_cohens_d_paired(arr_a, arr_b)),
        "ci": (ci_low, ci_high),
        "bootstrap_ci": (boot_low, boot_high),
    }


def _welch_test(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
) -> Dict:

    result = stats.ttest_ind(
        arr_a,
        arr_b,
        equal_var=False,
        nan_policy="omit",
    )

    mean_a = np.mean(arr_a)
    mean_b = np.mean(arr_b)

    var_a = np.var(arr_a, ddof=1)
    var_b = np.var(arr_b, ddof=1)

    n_a = len(arr_a)
    n_b = len(arr_b)

    mean_diff = mean_a - mean_b

    # Welch-Satterthwaite standard error
    se = np.sqrt(var_a / n_a + var_b / n_b)

    # Welch-Satterthwaite degrees of freedom
    df_num = (var_a / n_a + var_b / n_b) ** 2

    df_den = (
        ((var_a / n_a) ** 2) / (n_a - 1)
        + ((var_b / n_b) ** 2) / (n_b - 1)
    )

    df = df_num / df_den

    # Confidence interval for difference in means
    t_crit = stats.t.ppf(0.975, df)

    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    # Bootstrap CI for difference of means (independent resampling)
    rng = np.random.default_rng(42)
    boot_diffs = np.array([
        np.mean(rng.choice(arr_a, size=n_a, replace=True))
        - np.mean(rng.choice(arr_b, size=n_b, replace=True))
        for _ in range(10_000)
    ])
    boot_low = float(np.percentile(boot_diffs, 2.5))
    boot_high = float(np.percentile(boot_diffs, 97.5))

    return {
        "test": "welch",
        "n_a": n_a,
        "n_b": n_b,
        "statistic": float(result.statistic),   # type: ignore
        "pvalue": float(result.pvalue),         # type: ignore
        "effect_size": float(_cohens_d_independent(arr_a, arr_b)),
        "ci": (float(ci_low), float(ci_high)),
        "bootstrap_ci": (boot_low, boot_high),
    }

# ============================================================================
# Reporting
# ============================================================================

def _print_summary(
    metric: str,
    values_a: np.ndarray,
    values_b: np.ndarray,
    t_result: Dict,
    w_result: Dict,
    note: str,
) -> None:

    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)

    diff = mean_a - mean_b

    std_a = np.std(values_a, ddof=1) if len(values_a) > 1 else 0.0
    std_b = np.std(values_b, ddof=1) if len(values_b) > 1 else 0.0

    result = t_result

    ci_low, ci_high = result["ci"]
    boot_low, boot_high = result["bootstrap_ci"]

    print(f"{metric}:")
    print(f"  test: {result['test']} ({note})")

    if result["test"] == "paired":
        print(f"  matched runs: {result['n']}")
    else:
        print(f"  model A runs: {result['n_a']}")
        print(f"  model B runs: {result['n_b']}")

    print(f"  model A mean: {mean_a:.6f}")
    print(f"  model A std:  {std_a:.6f}")

    print(f"  model B mean: {mean_b:.6f}")
    print(f"  model B std:  {std_b:.6f}")

    print(f"  mean difference (A - B): {diff:.6f}")

    # Replace the three hardcoded-label lines:
    t_label = "paired t-test" if t_result["test"] == "paired" else "Welch's t-test"
    np_label = "Wilcoxon" if w_result["test"] == "wilcoxon" else "Mann-Whitney U"

    print(f"  {t_label} p-value:  {t_result['pvalue']:.6g}")
    print(f"  {np_label} p-value: {w_result['pvalue']:.6g}")
    print(f"  {np_label} statistic: {w_result['statistic']:.6f}")

    print(f"  effect size (Cohen's d): {result['effect_size']:.6f}")

    print(
        f"  95% CI of difference: "
        f"[{ci_low:.6f}, {ci_high:.6f}]"
    )

    print(
        f"  Bootstrap 95% CI: "
        f"[{boot_low:.6f}, {boot_high:.6f}]"
    )

    print()


# ============================================================================
# Main comparison
# ============================================================================

def _compare_metric(
    model_a: List[ModelRun],
    model_b: List[ModelRun],
    metric: str,
) -> None:

    metric_a = _build_metric_map(model_a, metric)
    metric_b = _build_metric_map(model_b, metric)

    if not metric_a or not metric_b:
        print(f"{metric}: missing values, skipping")
        return

    # Aggregate folds within each seed → independent observations
    agg_a = _aggregate_by_seed(metric_a)
    agg_b = _aggregate_by_seed(metric_b)

    common_seeds = sorted(
        agg_a.keys() & agg_b.keys(),
        key=lambda s: (s is None, s),
    )

    # ------------------------------------------------------------------
    # Paired test on per-seed means
    # ------------------------------------------------------------------

    if len(common_seeds) >= 2:

        arr_a = np.asarray([agg_a[s] for s in common_seeds], dtype=float)
        arr_b = np.asarray([agg_b[s] for s in common_seeds], dtype=float)

        t_result = _paired_test(arr_a, arr_b)
        w_result = _wilcoxon_test(arr_a, arr_b)

        _print_summary(
            metric=metric,
            values_a=arr_a,
            values_b=arr_b,
            t_result=t_result,
            w_result=w_result,
            note=f"paired on {len(common_seeds)} seed-level means",
        )

        return

    # ------------------------------------------------------------------
    # Welch fallback — use Mann-Whitney (unpaired) not Wilcoxon
    # ------------------------------------------------------------------

    arr_a = np.asarray(list(agg_a.values()), dtype=float)
    arr_b = np.asarray(list(agg_b.values()), dtype=float)

    if len(arr_a) < 2 or len(arr_b) < 2:
        print(f"{metric}: insufficient data")
        return

    t_result = _welch_test(arr_a, arr_b)
    w_result = _mann_whitney_test(arr_a, arr_b)

    _print_summary(
        metric=metric,
        values_a=arr_a,
        values_b=arr_b,
        t_result=t_result,
        w_result=w_result,
        note="Welch fallback on seed-level means",
    )


# ============================================================================
# Entry point
# ============================================================================

def main(model_a: str, model_b: str) -> None:

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

    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(f"Model A runs: {len(runs_a)}")
    print(f"Model B runs: {len(runs_b)}")

    bonferroni_alpha = 0.05 / len(METRICS)
    print(
        f"\nBonferroni-corrected significance threshold: "
        f"{bonferroni_alpha:.4f}  (α=0.05 / {len(METRICS)} metrics)\n"
    )

    print()

    for metric in METRICS:
        _compare_metric(runs_a, runs_b, metric)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compare two model log sets with statistical tests "
            "on F1 and ROC-AUC."
        )
    )

    parser.add_argument(
        "model_a",
        nargs="?",
        help=(
            "Path to model A log file or directory "
            "(default: logs/folds/pretrained)"
        ),
        default=str(HERE / "pretrained"),
    )

    parser.add_argument(
        "model_b",
        nargs="?",
        help=(
            "Path to model B log file or directory "
            "(default: logs/folds/scratch)"
        ),
        default=str(HERE / "scratch"),
    )

    args = parser.parse_args()

    main(args.model_a, args.model_b)