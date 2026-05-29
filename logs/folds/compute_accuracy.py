"""Compute per-seed accuracy summaries from fold evaluation logs.

The default input directory is ``logs/folds/scratch``. Each ``.out`` file is
expected to contain multiple fold records for a single seed. The script
averages ``eval_accuracy`` across folds for each seed, then reports the best
and worst seed according to that per-seed mean.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import parse_folds


HERE = Path(__file__).parent
DEFAULT_DIR = HERE / "mice" / "pretrained"


def _load_accuracy_by_seed(log_dir: Path) -> Dict[int, float]:
	"""Return seed -> mean accuracy over all folds found in log_dir."""

	items = parse_folds.data_from_files(str(log_dir))
	if not items:
		raise FileNotFoundError(f"No .out files found in {log_dir}")

	buckets: dict[int, list[float]] = defaultdict(list)

	for filename, line in items:
		seed = parse_folds.seed_from_filename(filename)
		data = parse_folds.parse_fold(line, seed)

		if not data:
			continue

		value = data.get("eval_accuracy")
		if value is None:
			continue

		buckets[seed].append(float(value))

	if not buckets:
		raise ValueError(f"No eval_accuracy values found in {log_dir}")

	return {
		seed: float(np.mean(values))
		for seed, values in sorted(buckets.items())
	}


def _best_and_worst(acc_by_seed: Dict[int, float]) -> Tuple[Tuple[int, float], Tuple[int, float]]:
	"""Return ((best_seed, best_accuracy), (worst_seed, worst_accuracy))."""

	best_seed = max(acc_by_seed, key=acc_by_seed.get)
	worst_seed = min(acc_by_seed, key=acc_by_seed.get)
	return (best_seed, acc_by_seed[best_seed]), (worst_seed, acc_by_seed[worst_seed])


def _print_report(acc_by_seed: Dict[int, float]) -> None:
	best, worst = _best_and_worst(acc_by_seed)

	print("Per-seed mean accuracies:")
	for seed, accuracy in acc_by_seed.items():
		print(f"  seed {seed}: {accuracy:.6f}")

	print()
	print(f"Best seed:  {best[0]} ({best[1]:.6f})")
	print(f"Worst seed: {worst[0]} ({worst[1]:.6f})")
	print(f"Mean over seeds: {float(np.mean(list(acc_by_seed.values()))):.6f}")


def main(log_dir: Optional[str] = None) -> Dict[int, float]:
	"""Parse fold logs and return seed -> mean accuracy."""

	log_dir_path = Path(log_dir) if log_dir is not None else DEFAULT_DIR
	acc_by_seed = _load_accuracy_by_seed(log_dir_path)
	_print_report(acc_by_seed)
	return acc_by_seed


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Average eval_accuracy by seed from fold logs."
	)
	parser.add_argument(
		"--logs-dir",
		type=str,
		default=str(DEFAULT_DIR),
		help="Directory containing the fold .out files.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	main(log_dir=args.logs_dir)
