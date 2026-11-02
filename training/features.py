"""Handcrafted feature extraction + disk cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "mean",
    "std",
    "p50",
    "p75",
    "p95",
    "tex",
    "bright_20",
    "bright_40",
    "bright_60",
    "bright_80",
    "bright40_L",
    "mean_L",
    "bright40_C",
    "mean_C",
    "bright40_R",
    "mean_R",
    "mean_top",
    "mean_bot",
    "bright40_top",
    "bright40_bot",
]


def load_feature_cache(out_dir: Path):
    out_dir = Path(out_dir)
    data = np.load(out_dir / "frame_features.npz", allow_pickle=True)
    X = data["X"]
    names = data["names"].tolist() if "names" in data.files else list(FEATURE_NAMES)
    names = [str(n) for n in names]
    meta = pd.read_csv(out_dir / "frame_features_meta.csv")
    assert len(meta) == len(X)
    return X, meta, names
