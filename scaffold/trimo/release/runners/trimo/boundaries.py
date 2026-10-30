"""ID, target and metric boundaries; importing this module does not import PyG."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def id_list(values, name):
    values = list(values)
    if any(type(mid) is not int for mid in values) or len(values) != len(set(values)):
        raise ValueError(f"{name}: molecule IDs must be unique integers")
    return values


def select_rows(cache_ids, split, retained_ids):
    """Exact cohort coverage; cache may be a superset (official QM9)."""
    ids = np.asarray(cache_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("Cache IDs must be one-dimensional integers")
    cache = id_list(ids.tolist(), "cache")
    retained = set(id_list(retained_ids, "retained"))
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Split must contain train, val, test ID lists")
    parts = {part: id_list(values, part) for part, values in split.items()}
    flat = [mid for values in parts.values() for mid in values]
    if len(flat) != len(set(flat)):
        raise ValueError("Split ID overlap")
    if set(flat) != retained or not retained <= set(cache):
        raise ValueError("Split must cover the retained cohort exactly, inside cache IDs")
    if len(parts["train"]) < 2 or not parts["val"]:
        raise ValueError("Need at least two train IDs and nonempty validation")
    row_for_id = {mid: row for row, mid in enumerate(cache)}
    # Test rows are intentionally not returned to the training interface.
    return {part: np.asarray([row_for_id[mid] for mid in parts[part]], dtype=np.int64)
            for part in ("train", "val")}


def training_scaling(selected_train_targets):
    values = np.asarray(selected_train_targets, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Training targets must be a finite vector")
    mean, std = float(values.mean()), float(values.std(ddof=0))
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise ValueError("Training standard deviation must be finite and positive")
    return {"mean": mean, "std": std, "ddof": 0, "n_fit": len(values)}


class RawMetrics:
    """Sum per-molecule errors, never average per-batch metric means."""
    def __init__(self, expected_ids):
        self.expected = set(id_list(expected_ids, "metric IDs"))
        self.seen = set()
        self.absolute = self.squared = 0.0

    def update(self, ids, prediction, target):
        ids = id_list(ids, "batch IDs")
        prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
        if prediction.shape != (len(ids),) or target.shape != prediction.shape:
            raise ValueError("One scalar prediction/target per molecule; no broadcasting")
        if not set(ids) <= self.expected or self.seen & set(ids):
            raise ValueError("Unexpected/repeated metric molecule ID")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise FloatingPointError("Nonfinite prediction/target")
        error = prediction - target
        self.absolute += float(np.abs(error).sum())
        self.squared += float(np.square(error).sum())
        if not np.isfinite([self.absolute, self.squared]).all():
            raise FloatingPointError("Nonfinite accumulated raw-unit errors")
        self.seen.update(ids)

    def result(self):
        if self.seen != self.expected or not self.seen:
            raise ValueError("Metric pass did not cover its IDs exactly once")
        n = len(self.seen)
        return {"n": n, "mae": self.absolute / n, "rmse": float(np.sqrt(self.squared / n))}


def check_binding(saved, expected):
    if saved != expected:
        raise ValueError("Resume source/config/split/cache/cohort binding mismatch")
