"""B5 pass-level mean pooling and B6 LSTM on handcrafted feature sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import cls3, cls5, progress


def build_pass_table(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta["feat_idx"] = np.arange(len(meta))
    return meta


def iter_pass_groups(meta: pd.DataFrame):
    for (grind, pass_id), gdf in meta.groupby(["grind", "pass_id"], sort=True):
        gdf = gdf.sort_values("frame_in_pass")
        yield grind, pass_id, gdf


def nested_part_from_grind(grind: int) -> str:
    """Nested interleaved parts: train/val from even stages; odd stages = test.

    train: g ≡ 0 (mod 4), val: g ≡ 2 (mod 4), test: g odd.
    """
    g = int(grind)
    if g % 2 == 1:
        return "test"
    if g % 4 == 0:
        return "train"
    return "val"


def annotate_nested_split(meta: pd.DataFrame) -> pd.DataFrame:
    """Add/overwrite split_nested from grind indices."""
    out = meta.copy()
    out["split_nested"] = out["grind"].map(nested_part_from_grind)
    return out


def pass_split_tag(gdf: pd.DataFrame, split: str) -> str:
    """Majority vote of frame split tags; ignore 'ignore' for spatial."""
    if split == "interleaved":
        return str(gdf["split_interleaved"].iloc[0])
    if split == "nested":
        if "split_nested" in gdf.columns:
            return str(gdf["split_nested"].iloc[0])
        return nested_part_from_grind(int(gdf["grind"].iloc[0]))
    if split == "random":
        return str(gdf["split_random"].iloc[0])
    tags = gdf["split_spatial"].value_counts()
    for t in ("train", "test"):
        if t in tags.index:
            return t
    return "ignore"


def make_pass_features(X: np.ndarray, meta: pd.DataFrame, split: str):
    """Aggregate mean/std/max/min over each pass; return Xpass, labels, tags, keys."""
    rows_x, y3, y5, yp, tags, keys = [], [], [], [], [], []
    for grind, pass_id, gdf in iter_pass_groups(meta):
        tag = pass_split_tag(gdf, split)
        if tag == "ignore":
            continue
        idx = gdf["feat_idx"].to_numpy()
        feats = X[idx]
        agg = np.concatenate(
            [feats.mean(0), feats.std(0), feats.max(0), feats.min(0)]
        ).astype(np.float32)
        rows_x.append(agg)
        y3.append(cls3(int(grind)))
        y5.append(cls5(int(grind)))
        yp.append(progress(int(grind)))
        tags.append(tag)
        keys.append((int(grind), int(pass_id)))
    return (
        np.stack(rows_x),
        np.asarray(y3),
        np.asarray(y5),
        np.asarray(yp, dtype=np.float64),
        np.asarray(tags),
        keys,
    )
