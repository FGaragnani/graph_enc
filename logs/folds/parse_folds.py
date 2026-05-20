from dataclasses import dataclass
import os
import ast
import re


@dataclass
class FoldData:
    fold: int
    eval_samples: int
    seed: int
    eval_loss: float
    eval_accuracy: float
    eval_f1: float
    eval_roc_auc: float

def divide_by_line(data: str) -> list[str]:
    return data.splitlines()


def seed_from_filename(filename: str) -> int:
    m = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    return int(m.group(1)) if m else None


def parse_fold(line: str, seed: int) -> dict:
    # Parse line format: "Fold k: {...}"
    parts = line.split("Fold ", 1)
    if len(parts) != 2:
        return {}
    
    parts = parts[1].split(": ", 1)
    fold_num = int(parts[0].strip())
    data_str = parts[1]
    
    # Parse the dictionary string (Python literal with single quotes)
    data = ast.literal_eval(data_str)
    data['fold'] = fold_num
    # include seed (if available) from filename
    data['seed'] = seed

    data.pop('eval_runtime', None)
    data.pop('eval_samples_per_second', None)
    data.pop('eval_steps_per_second', None)
    data.pop('epoch', None)

    return data
    
def data_from_files(dir: str = "logs/folds/") -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for filename in os.listdir(dir):
        if filename.endswith(".out"):
            path = os.path.join(dir, filename)
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append((filename, line))
    return items

def get_folds_data(items: list[tuple[str, str]]) -> list[FoldData]:
    folds = []
    for filename, line in items:
        seed = seed_from_filename(filename)
        fold_data = parse_fold(line, seed)
        if fold_data:
            folds.append(FoldData(**fold_data))
    return folds

if __name__ == "__main__":
    items = data_from_files()
    folds = get_folds_data(items)

