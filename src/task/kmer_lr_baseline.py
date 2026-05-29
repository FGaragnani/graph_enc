"""k-mer logistic regression baseline for promoter vs. enhancer classification.

This baseline plugs into the *same* evaluation protocol as the transformer
(``src/task/discriminative.py``):

* same chromosome-based train/test split,
* same greedy 10-fold cross-validation over chromosomes,
* same per-fold class rebalancing to ``target_enhancer_fraction`` (default 0.5),
* same 7 seeds (42..48).

Features
--------
For each sequence we build a single feature vector by concatenating the
*normalized* k-mer frequency vectors for ``k=3`` and ``k=6`` (64 + 4096 = 4160
dims). Frequencies are normalized by the number of valid k-mer positions, so
variable-length sequences are handled directly.

Long sequences (> ``chunk_size_bases`` nt, default 2000) are split into
non-overlapping chunks exactly like the transformer's data collator
(``range(0, len, chunk_size)``, capped at ``max_chunks_per_sample``), the k-mer
frequency vector is computed per chunk, and the per-chunk vectors are
mean-pooled (equal weight) before classification -- matching the
``ChunkAveragedCLSClassifier`` semantics. Promoters are exactly 2000 nt, so they
yield a single chunk.

Classifier
----------
``sklearn.linear_model.LogisticRegression(solver="liblinear", C=1.0,
random_state=seed)``. No tuning -- this is a baseline. The run ``seed`` controls
the solver's ``random_state``; by default it is *also* used as the
``balance_seed`` so that, like the existing per-seed transformer runs
(``--balance_seed ${seed}`` in ``scripts/finetune_pe.sh``), each seed resamples
the balanced set independently. This is what produces non-degenerate
seed-level variance; pass ``--fixed-balance-seed`` to instead hold the split
fixed and let only ``random_state`` vary (variance will then be ~0).

Outputs
-------
* A pandas DataFrame with columns ``[seed, fold, model, F1, ROC-AUC, accuracy]``
  (the in-memory form of the existing results), saved to ``<output_dir>/
  kmer_lr_results.csv``.
* One ``<seed>.out`` log per seed under ``<output_dir>``, written in the exact
  ``Fold k: {dict}`` format consumed by ``logs/folds/parse_folds.py`` -- so the
  existing comparison scripts (paired t-test / Wilcoxon / TOST) can be run
  directly, e.g.::

      python logs/folds/compare_model_logs_ttest.py logs/folds/pretrained <output_dir>
      python logs/folds/compare_model_logs_ttest_r.py logs/folds/pretrained <output_dir>

* A printed summary table of seed-level mean +/- 95% t-interval CI for F1,
  ROC-AUC and accuracy (n = number of seeds), matching the report tables.

Example
-------
    python src/task/kmer_lr_baseline.py \
        --dataset_dir scripts/datasets \
        --data_path data/ \
        --output_dir logs/folds/kmer_lr
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from graph_enc.src.data.en_prom_dataset import PromoterEnhancerDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("kmer_lr")

DEFAULT_SEEDS = (42, 43, 44, 45, 46, 47, 48)
_BASE4 = {"A": 0, "C": 1, "G": 2, "T": 3}


# ===========================================================================
# Feature extraction
# ===========================================================================

def _kmer_block_sizes(ks: Sequence[int]) -> List[int]:
    return [4 ** k for k in ks]


def _chunk_kmer_counts(chunk: str, k: int, n_features: int) -> np.ndarray:
    """Raw k-mer counts for one chunk; windows containing non-ACGT are skipped."""
    counts = np.zeros(n_features, dtype=np.float64)
    code = 0
    valid = 0  # number of consecutive valid (ACGT) bases ending at current pos
    mask = n_features - 1  # n_features == 4**k, so this masks to the low 2k bits
    for base in chunk:
        digit = _BASE4.get(base)
        if digit is None:
            valid = 0
            code = 0
            continue
        code = ((code << 2) | digit) & mask
        valid += 1
        if valid >= k:
            counts[code] += 1.0
    return counts


def sequence_to_feature_vector(
    sequence: str,
    ks: Sequence[int],
    chunk_size_bases: int,
    max_chunks_per_sample: Optional[int],
) -> np.ndarray:
    """Concatenated, length-normalized, chunk-mean-pooled k-mer frequencies.

    For each chunk and each ``k`` the raw counts are normalized by the number of
    valid k-mer positions in that chunk (so each k-block is a frequency
    distribution), then the per-chunk vectors are averaged with equal weight.
    """
    sequence = sequence.strip().upper()
    if not sequence:
        sequence = "N"

    block_sizes = _kmer_block_sizes(ks)
    total_dim = sum(block_sizes)

    # Non-overlapping chunks, mirroring the transformer's collator.
    chunks = [
        sequence[i : i + chunk_size_bases]
        for i in range(0, len(sequence), chunk_size_bases)
    ] or ["N"]
    if max_chunks_per_sample is not None:
        chunks = chunks[:max_chunks_per_sample]

    pooled = np.zeros(total_dim, dtype=np.float64)
    for chunk in chunks:
        offset = 0
        chunk_vec = np.zeros(total_dim, dtype=np.float64)
        for k, n_features in zip(ks, block_sizes):
            counts = _chunk_kmer_counts(chunk, k, n_features)
            total = counts.sum()
            if total > 0:
                chunk_vec[offset : offset + n_features] = counts / total
            offset += n_features
        pooled += chunk_vec

    pooled /= len(chunks)  # equal-weight mean pool over chunks
    return pooled.astype(np.float32)


def build_feature_matrix(
    base_dataset: PromoterEnhancerDataset,
    ks: Sequence[int],
    chunk_size_bases: int,
    max_chunks_per_sample: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute features + labels once for every base sample (fold/seed agnostic)."""
    n = len(base_dataset)
    total_dim = sum(_kmer_block_sizes(ks))
    X = np.zeros((n, total_dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)

    log_every = max(1, n // 20)
    for idx in range(n):
        item = base_dataset.data[idx]
        sequence = base_dataset[idx]  # materializes enhancer/promoter sequence
        X[idx] = sequence_to_feature_vector(
            sequence, ks, chunk_size_bases, max_chunks_per_sample
        )
        y[idx] = 1 if item.is_enhancer() else 0
        if (idx + 1) % log_every == 0:
            logger.info("Featurized %d/%d sequences", idx + 1, n)

    logger.info("Feature matrix: %s (%d positives, %d negatives)",
                X.shape, int(y.sum()), int((y == 0).sum()))
    return X, y


# ===========================================================================
# Splits (mirrors src/task/discriminative.py, do_eval + perform_kfold path)
# ===========================================================================

def _rebalance_indices(
    idx_list: List[int],
    base_dataset: PromoterEnhancerDataset,
    target_enhancer_fraction: float,
    balance_seed: int,
    fold_id: int,
) -> List[int]:
    items = [base_dataset.data[i] for i in idx_list]
    enhancers = [i for i, item in zip(idx_list, items) if item.is_enhancer()]
    promoters = [i for i, item in zip(idx_list, items) if item.is_promoter()]

    if not enhancers or not promoters:
        logger.warning(
            "Fold %d: cannot rebalance (enhancers=%d, promoters=%d)",
            fold_id, len(enhancers), len(promoters),
        )
        return idx_list

    rng = random.Random(balance_seed)
    target = target_enhancer_fraction
    desired_enhancers = int(round((target / (1.0 - target)) * len(promoters)))

    if desired_enhancers <= len(enhancers):
        selected_enhancers = rng.sample(enhancers, desired_enhancers)
        selected_promoters = promoters
    else:
        desired_promoters = int(round(((1.0 - target) / target) * len(enhancers)))
        selected_enhancers = enhancers
        selected_promoters = rng.sample(promoters, desired_promoters)

    balanced = selected_enhancers + selected_promoters
    rng.shuffle(balanced)
    return balanced


def build_folds(
    base_dataset: PromoterEnhancerDataset,
    num_folds: int,
    target_enhancer_fraction: float,
    balance_seed: int,
) -> List[Tuple[List[int], List[int]]]:
    """Greedy chromosome-balanced K folds, then per-fold rebalancing.

    Returns a list of ``(eval_idx, train_idx)`` tuples, identical in
    construction to discriminative.py's eval + perform_kfold branch.
    """
    chr_to_idx: Dict[int, List[int]] = defaultdict(list)
    for idx, item in enumerate(base_dataset.data):
        chr_to_idx[item.get_chr_idx()].append(idx)

    sorted_chrs = sorted(chr_to_idx.keys())
    chr_sizes = {chr_id: len(chr_to_idx[chr_id]) for chr_id in sorted_chrs}
    sorted_chrs_by_size = sorted(chr_sizes.items(), key=lambda x: x[1], reverse=True)

    num_folds = min(num_folds, len(sorted_chrs))
    fold_chrs: List[List[int]] = [[] for _ in range(num_folds)]

    # Greedy: assign each chromosome to the currently-smallest fold.
    for chr_id, _size in sorted_chrs_by_size:
        min_fold = min(
            range(num_folds),
            key=lambda i: sum(chr_sizes[c] for c in fold_chrs[i]),
        )
        fold_chrs[min_fold].append(chr_id)

    raw_folds: List[Tuple[List[int], List[int]]] = []
    for fold_i in range(num_folds):
        eval_chrs = fold_chrs[fold_i]
        train_chrs = [c for i in range(num_folds) if i != fold_i for c in fold_chrs[i]]
        eval_idx = [idx for chr_id in eval_chrs for idx in chr_to_idx[chr_id]]
        train_idx = [idx for chr_id in train_chrs for idx in chr_to_idx[chr_id]]
        if eval_idx and train_idx:
            raw_folds.append((eval_idx, train_idx))

    folds = [
        (
            _rebalance_indices(eval_idx, base_dataset, target_enhancer_fraction, balance_seed, fold_idx),
            _rebalance_indices(train_idx, base_dataset, target_enhancer_fraction, balance_seed, fold_idx),
        )
        for fold_idx, (eval_idx, train_idx) in enumerate(raw_folds)
    ]
    return folds


# ===========================================================================
# Training / evaluation
# ===========================================================================

def evaluate_fold(
    X: np.ndarray,
    y: np.ndarray,
    eval_idx: List[int],
    train_idx: List[int],
    seed: int,
    C: float,
    solver: str,
    max_iter: int,
) -> Dict[str, float]:
    clf = LogisticRegression(
        solver=solver,
        C=C,
        random_state=seed,
        max_iter=max_iter,
    )
    clf.fit(X[train_idx], y[train_idx])

    y_true = y[eval_idx]
    probs = clf.predict_proba(X[eval_idx])[:, 1]
    preds = (probs >= 0.5).astype(np.int64)

    accuracy = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, average="binary"))
    if len(np.unique(y_true)) >= 2:
        roc_auc = float(roc_auc_score(y_true, probs))
    else:
        logger.warning("Only one class in eval fold; ROC-AUC undefined.")
        roc_auc = float("nan")
    loss = float(log_loss(y_true, probs, labels=[0, 1]))

    return {
        "eval_loss": loss,
        "eval_accuracy": accuracy,
        "eval_f1": f1,
        "eval_roc_auc": roc_auc,
        "eval_samples": int(len(eval_idx)),
    }


def run_baseline(
    base_dataset: PromoterEnhancerDataset,
    X: np.ndarray,
    y: np.ndarray,
    seeds: Sequence[int],
    num_folds: int,
    target_enhancer_fraction: float,
    fixed_balance_seed: Optional[int],
    model_name: str,
    C: float,
    solver: str,
    max_iter: int,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for seed in seeds:
        balance_seed = seed if fixed_balance_seed is None else fixed_balance_seed
        folds = build_folds(base_dataset, num_folds, target_enhancer_fraction, balance_seed)
        logger.info("Seed %d: %d folds (balance_seed=%d)", seed, len(folds), balance_seed)

        for fold_idx, (eval_idx, train_idx) in enumerate(folds):
            metrics = evaluate_fold(
                X, y, eval_idx, train_idx, seed, C, solver, max_iter
            )
            logger.info(
                "Seed %d Fold %d: acc=%.4f f1=%.4f roc_auc=%.4f",
                seed, fold_idx, metrics["eval_accuracy"],
                metrics["eval_f1"], metrics["eval_roc_auc"],
            )
            records.append(
                {
                    "seed": seed,
                    "fold": fold_idx,
                    "model": model_name,
                    "F1": metrics["eval_f1"],
                    "ROC-AUC": metrics["eval_roc_auc"],
                    "accuracy": metrics["eval_accuracy"],
                    # kept for .out export, not part of the canonical columns
                    "_eval_loss": metrics["eval_loss"],
                    "_eval_samples": metrics["eval_samples"],
                }
            )

    return pd.DataFrame.from_records(records)


# ===========================================================================
# Outputs
# ===========================================================================

def write_out_logs(df: pd.DataFrame, output_dir: Path, model_name: str) -> None:
    """Write one <seed>.out per seed in the parse_folds `Fold k: {dict}` format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

    for seed, group in df.groupby("seed"):
        lines = []
        for _, row in group.sort_values("fold").iterrows():
            fold_dict = {
                "eval_loss": float(row["_eval_loss"]),
                "eval_accuracy": float(row["accuracy"]),
                "eval_f1": float(row["F1"]),
                "eval_roc_auc": float(row["ROC-AUC"]),
                "eval_samples": int(row["_eval_samples"]),
            }
            lines.append(
                f"{timestamp} - INFO - {model_name} - "
                f"Fold {int(row['fold'])}: {fold_dict!r}"
            )
        out_path = output_dir / f"{int(seed)}.out"
        out_path.write_text("\n".join(lines) + "\n")
        logger.info("Wrote %s (%d folds)", out_path, len(lines))


def _t_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """Classical t-interval over per-seed means (matches logs/folds/ci_utils.t_interval)."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) - 1 <= 0:
        return mean, (mean, mean)
    se = float(stats.sem(arr))
    t = stats.t.ppf(1 - (1 - confidence) / 2, df=len(arr) - 1)
    return mean, (mean - t * se, mean + t * se)


def print_summary(df: pd.DataFrame, model_name: str) -> None:
    """Seed-level mean +/- 95% t-interval CI, matching the report tables."""
    metrics = ["F1", "ROC-AUC", "accuracy"]
    # Average folds within each seed -> independent seed-level observations.
    seed_means = df.groupby("seed")[metrics].mean()
    n = len(seed_means)

    print()
    print("=" * 72)
    print(f"k-mer LR baseline ('{model_name}') -- seed-level summary "
          f"(n = {n} seeds, {df['fold'].nunique()}-fold CV each)")
    print("=" * 72)
    header = f"{'Metric':<10} {'Mean':>10}   {'95% t-interval CI':>26}"
    print(header)
    print("-" * 72)
    for metric in metrics:
        mean, (lo, hi) = _t_interval(seed_means[metric].to_numpy())
        print(f"{metric:<10} {mean:>10.4f}   [{lo:>10.4f}, {hi:>10.4f}]")
    print("=" * 72)


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="k-mer logistic regression baseline (promoter vs. enhancer).",
    )
    parser.add_argument("--dataset_dir", default="scripts/datasets",
                        help="Directory with promoters/enhancers files.")
    parser.add_argument("--data_path", default=None,
                        help="Genome directory used to materialize sequences (required).")
    parser.add_argument("--is_human", default="true",
                        choices=["true", "false"], help="Human (true) or mouse (false) data.")
    parser.add_argument("--output_dir", default="logs/folds/kmer_lr",
                        help="Where to write <seed>.out logs and the results CSV.")
    parser.add_argument("--model_name", default="kmer_lr",
                        help="Model label used in the DataFrame and logs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seeds (default 42..48). Each controls LR random_state.")
    parser.add_argument("--ks", type=int, nargs="+", default=[3, 6],
                        help="k values to concatenate (default 3 6).")
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--target_enhancer_fraction", type=float, default=0.5)
    parser.add_argument("--chunk_size_bases", type=int, default=2000)
    parser.add_argument("--max_chunks_per_sample", type=int, default=10,
                        help="Cap on chunks per sequence (matches finetune_pe.sh). "
                             "Use 0 for no cap.")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", default="liblinear")
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--fixed-balance-seed", dest="fixed_balance_seed",
                        type=int, default=None,
                        help="If set, hold the split/rebalancing fixed at this seed "
                             "for all runs (only random_state varies). Default: tie "
                             "balance_seed to the run seed, like the transformer runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_path is None:
        raise SystemExit(
            "--data_path (genome directory) is required: enhancer and promoter "
            "sequences are materialized from chromosome FASTA files."
        )

    max_chunks = None if args.max_chunks_per_sample in (0, None) else args.max_chunks_per_sample

    logger.info("Loading dataset (no rebalancing at load; folds rebalance per seed)...")
    base_dataset = PromoterEnhancerDataset(
        dir=args.dataset_dir,
        genome_data_path=args.data_path,
        target_enhancer_fraction=None,
        apply_rebalancing=False,
        is_human=(args.is_human == "true"),
        promoter_window_size=args.chunk_size_bases,
    )

    logger.info("Building k-mer feature matrix (k=%s)...", args.ks)
    X, y = build_feature_matrix(base_dataset, args.ks, args.chunk_size_bases, max_chunks)

    df = run_baseline(
        base_dataset=base_dataset,
        X=X,
        y=y,
        seeds=args.seeds,
        num_folds=args.num_folds,
        target_enhancer_fraction=args.target_enhancer_fraction,
        fixed_balance_seed=args.fixed_balance_seed,
        model_name=args.model_name,
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Canonical results DataFrame (same columns as the existing results).
    canonical = df[["seed", "fold", "model", "F1", "ROC-AUC", "accuracy"]]
    csv_path = output_dir / "kmer_lr_results.csv"
    canonical.to_csv(csv_path, index=False)
    logger.info("Saved results DataFrame to %s", csv_path)

    write_out_logs(df, output_dir, args.model_name)
    print_summary(df, args.model_name)

    print(f"\nLogs written to {output_dir}/. Compare against an existing model with, e.g.:")
    print(f"  python logs/folds/compare_model_logs_ttest.py logs/folds/pretrained {output_dir}")
    print(f"  python logs/folds/compare_model_logs_ttest_r.py logs/folds/pretrained {output_dir}")


if __name__ == "__main__":
    main()
