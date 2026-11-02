"""Frame dataset helpers for CNN baselines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .labels import cls3, cls5, progress


def default_transform(train: bool = False, height: int = 128, width: int = 768):
    ops = [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if train:
        ops.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(ops)


def load_frames_meta(frames_csv: Path) -> pd.DataFrame:
    return pd.read_csv(frames_csv)


class FrameDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, split: str, part: str, train: bool = False, height=128, width=768):
        if split == "interleaved":
            col = "split_interleaved"
        elif split == "nested":
            if "split_nested" not in meta.columns:
                from .temporal import annotate_nested_split

                meta = annotate_nested_split(meta)
            col = "split_nested"
        else:
            col = "split_spatial"
        self.meta = meta[meta[col] == part].reset_index(drop=True)
        if len(self.meta) == 0:
            raise ValueError(f"empty dataset {split}/{part}")
        self.tf = default_transform(train=train, height=height, width=width)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        r = self.meta.iloc[idx]
        img = Image.open(r.path).convert("RGB")
        x = self.tf(img)
        g = int(r.grind)
        return (
            x,
            cls3(g),
            cls5(g),
            torch.tensor(progress(g), dtype=torch.float32),
        )
