"""Shared evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def eval_classification(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def eval_regression(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(y_true, y_pred)[0, 1])
        # prefer scipy spearman if available
        try:
            from scipy.stats import spearmanr

            spearman = float(spearmanr(y_true, y_pred).correlation)
        except Exception:
            pass
    return {"mae": float(mean_absolute_error(y_true, y_pred)), "spearman": spearman}


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
