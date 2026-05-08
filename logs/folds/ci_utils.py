"""Confidence-interval utilities for F1 and ROC-AUC scores.

Provides:
- example-level bootstrap (`f1_bootstrap_ci`)
- hierarchical bootstrap across runs (`hierarchical_f1_bootstrap`)
- run-level bootstrap over per-run F1s (`run_level_bootstrap_f1`)
- example-level bootstrap (`roc_auc_bootstrap_ci`)
- hierarchical bootstrap across runs (`hierarchical_roc_auc_bootstrap`)
- t-interval over per-run F1s (`t_interval`)
- plotting helpers to visualize bootstrap distributions and CIs

Usage examples are in the `__main__` block.
"""
from typing import List, Tuple, Optional, Literal
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


def f1_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000,
                     alpha: float = 0.05, average: Literal['binary', 'micro', 'macro', 'weighted', 'samples'] = 'binary',
                     random_state: Optional[int] = None) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Example-level percentile bootstrap for F1.

    Returns (mean_estimate, (lo, hi), all_bootstrap_samples).
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[i] = f1_score(y_true[idx], y_pred[idx], average=average)
    lo, hi = np.percentile(samples, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(samples.mean()), (float(lo), float(hi)), samples


def hierarchical_f1_bootstrap(runs: List[Tuple[np.ndarray, np.ndarray]], n_boot: int = 2000,
                               alpha: float = 0.05, average: Literal['binary', 'micro', 'macro', 'weighted', 'samples'] = 'binary',
                               random_state: Optional[int] = None) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Hierarchical bootstrap: resample runs then examples within runs.

    `runs` should be a list of (y_true, y_pred) arrays for each run (seed/fold).
    """
    rng = np.random.default_rng(random_state)
    m = len(runs)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        # sample run indices with replacement
        run_indices = rng.integers(0, m, m)
        y_true_parts = []
        y_pred_parts = []
        for ridx in run_indices:
            yt, yp = runs[ridx]
            if len(yt) == 0:
                continue
            idx = rng.integers(0, len(yt), len(yt))
            y_true_parts.append(yt[idx])
            y_pred_parts.append(yp[idx])
        if not y_true_parts:
            samples[i] = np.nan
            continue
        f1s = []
        for ridx in run_indices:
            yt, yp = runs[ridx]
            idx = rng.integers(0, len(yt), len(yt))
            f1s.append(f1_score(yt[idx], yp[idx], average=average))

        samples[i] = np.mean(f1s)
    samples = samples[~np.isnan(samples)]
    lo, hi = np.percentile(samples, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(samples.mean()), (float(lo), float(hi)), samples


def roc_auc_bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 2000,
                        alpha: float = 0.05, random_state: Optional[int] = None) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Example-level percentile bootstrap for ROC-AUC.

    Returns (mean_estimate, (lo, hi), all_bootstrap_samples).
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            samples[i] = np.nan
            continue
        samples[i] = roc_auc_score(y_true[idx], y_score[idx])

    samples = samples[~np.isnan(samples)]
    lo, hi = np.percentile(samples, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(samples.mean()), (float(lo), float(hi)), samples


def hierarchical_roc_auc_bootstrap(
    runs: List[Tuple[np.ndarray, np.ndarray]],
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
):
    """Hierarchical bootstrap for ROC-AUC."""
    rng = np.random.default_rng(random_state)
    m = len(runs)
    samples = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        run_indices = rng.integers(0, m, m)

        aucs = []
        for ridx in run_indices:
            yt, ys = runs[ridx]
            idx = rng.integers(0, len(yt), len(yt))

            if len(np.unique(yt[idx])) < 2:
                continue

            aucs.append(roc_auc_score(yt[idx], ys[idx]))

        samples[i] = np.mean(aucs) if aucs else np.nan

    samples = samples[~np.isnan(samples)]
    lo, hi = np.percentile(samples, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(samples.mean()), (float(lo), float(hi)), samples


def run_level_bootstrap_f1(f1s: List[float], n_boot: int = 2000, alpha: float = 0.05,
                          random_state: Optional[int] = None) -> Tuple[float, Tuple[float, float], np.ndarray]:
    """Bootstrap over per-run F1 values (treat runs as atomic units).
    Useful when only per-run aggregated metrics are available.
    """
    rng = np.random.default_rng(random_state)
    arr = np.asarray(f1s, dtype=float)
    n = len(arr)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[i] = arr[idx].mean()
    lo, hi = np.percentile(samples, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(samples.mean()), (float(lo), float(hi)), samples


def t_interval(f1s: List[float], alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    """Classical t-interval over per-run F1s."""
    arr = np.asarray(f1s, dtype=float)
    mean = float(arr.mean())
    se = float(stats.sem(arr))
    if len(arr) - 1 <= 0:
        return mean, (mean, mean)
    t = stats.t.ppf(1 - alpha / 2, df=len(arr) - 1)
    return mean, (float(mean - t * se), float(mean + t * se))


def plot_bootstrap_distribution(samples: np.ndarray, ci: Tuple[float, float], title: Optional[str] = None,
                                save_path: Optional[str] = None, bins: int = 40) -> None:
    """Plot bootstrap distribution and mark CI and mean."""
    plt.figure(figsize=(7, 4))
    if _HAS_SEABORN:
        sns.histplot(samples, bins=bins, kde=True)
    else:
        plt.hist(samples, bins=bins, density=False, alpha=0.7)
    mean = float(np.mean(samples))
    lo, hi = ci
    plt.axvline(mean, color='C1', linestyle='-', label=f'mean={mean:.4f}')
    plt.axvline(lo, color='k', linestyle='--', label=f'CI lower={lo:.4f}')
    plt.axvline(hi, color='k', linestyle='--', label=f'CI upper={hi:.4f}')
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
