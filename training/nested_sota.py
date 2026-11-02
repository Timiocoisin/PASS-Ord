#!/usr/bin/env python3
"""Nested interleaved protocol: select on val, report once on test.

Parts (from grind index g):
  train: g % 4 == 0
  val:   g % 4 == 2
  test:  g odd

Selection (specialty checkpoints, ResNet Spearman ckpt, progress weights) uses
val only. Final experts are then refit on train∪val with frozen selection
choices and evaluated once on test. Baselines B5/B6/B3 follow the same fit/eval
split so SOTA is apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from .data import FrameDataset, load_frames_meta
from .features import load_feature_cache
from .labels import cls3, cls5, progress
from .metrics import eval_classification, eval_regression, save_json
from .temporal import (
    annotate_nested_split,
    build_pass_table,
    iter_pass_groups,
    make_pass_features,
    pass_split_tag,
)

ROOT = Path(__file__).resolve().parents[1]
PLATEAU_G = 94
SOFT_W = {"p3_lstm": 0.90, "p3_rf": 0.10, "p5_lstm": 0.25, "p5_rf": 0.75}
OUT = ROOT / "outputs/baselines/ours_nested"
TABLES = ROOT / "outputs/baselines/paper_tables"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_seg(path: Path):
    df = pd.read_csv(path)
    return {
        int(r.grind): np.array(
            [r.metal_ratio, r.metal_ratio_L, r.metal_ratio_C, r.metal_ratio_R], dtype=np.float32
        )
        for r in df.itertuples()
    }


class PassSeq(Dataset):
    def __init__(self, X, meta, seg, part, max_len=24, fit_parts=None):
        """If fit_parts is set (e.g. ('train','val')), keep those parts; else keep `part`."""
        self.seqs, self.segs = [], []
        self.y3, self.y5, self.yp, self.ypl, self.keys = [], [], [], [], []
        meta = build_pass_table(X, meta)
        want = set(fit_parts) if fit_parts is not None else {part}
        for grind, pass_id, gdf in iter_pass_groups(meta):
            tag = pass_split_tag(gdf, "nested")
            if tag not in want:
                continue
            idx = gdf["feat_idx"].to_numpy()
            feats = X[idx]
            if len(feats) > max_len:
                sel = np.linspace(0, len(feats) - 1, max_len, dtype=int)
                feats = feats[sel]
            g = int(grind)
            self.seqs.append(torch.from_numpy(feats.astype(np.float32)))
            self.segs.append(
                torch.from_numpy(seg.get(g, np.zeros(4, dtype=np.float32)).astype(np.float32))
            )
            self.y3.append(cls3(g))
            self.y5.append(cls5(g))
            self.yp.append(progress(g))
            self.ypl.append(1 if g >= PLATEAU_G else 0)
            self.keys.append((g, int(pass_id)))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return (
            self.seqs[i],
            self.segs[i],
            self.y3[i],
            self.y5[i],
            self.yp[i],
            self.ypl[i],
            self.keys[i][0],
            self.keys[i][1],
        )


def collate(batch):
    seqs, segs, y3, y5, yp, ypl, g, pid = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return (
        pad_sequence(seqs, batch_first=True),
        lens,
        torch.stack(segs),
        torch.tensor(y3, dtype=torch.long),
        torch.tensor(y5, dtype=torch.long),
        torch.tensor(yp, dtype=torch.float32),
        torch.tensor(ypl, dtype=torch.float32),
        torch.tensor(g, dtype=torch.long),
        torch.tensor(pid, dtype=torch.long),
    )


class TemporalExpert(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 160):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.15
        )
        dim = hidden * 2
        self.seg = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.drop = nn.Dropout(0.15)
        fused = dim + 32
        self.head_cls3 = nn.Linear(fused, 3)
        self.head_cls5 = nn.Linear(fused, 5)
        self.head_prog = nn.Linear(fused, 1)
        self.head_plat = nn.Linear(fused, 1)

    def forward(self, x, lens, seg):
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        f = self.drop(torch.cat([h, self.seg(seg)], dim=-1))
        return (
            self.head_cls3(f),
            self.head_cls5(f),
            self.head_prog(f).squeeze(-1),
            self.head_plat(f).squeeze(-1),
        )


def coral(logits, y, n):
    p = F.softmax(logits, dim=-1)
    loss = logits.new_zeros(())
    for k in range(n - 1):
        pk = p[:, k + 1 :].sum(-1).clamp(1e-6, 1 - 1e-6)
        loss = loss + F.binary_cross_entropy(pk, (y > k).float())
    return loss / (n - 1)


def compute_loss(o3, o5, op, opl, y3, y5, yp, ypl):
    return (
        2.6 * F.cross_entropy(o3, y3)
        + 1.2 * F.cross_entropy(o5, y5)
        + 2.0 * F.smooth_l1_loss(op, yp)
        + 0.40 * F.binary_cross_entropy_with_logits(opl, ypl)
        + 0.45 * coral(o3, y3, 3)
        + 0.25 * coral(o5, y5, 5)
    )


class MultiTaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.head_cls3 = nn.Linear(in_f, 3)
        self.head_cls5 = nn.Linear(in_f, 5)
        self.head_prog = nn.Linear(in_f, 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.head_cls3(f), self.head_cls5(f), self.head_prog(f).squeeze(-1)


def metrics_from_rows(rows):
    return {
        "cls3": eval_classification([r["t3"] for r in rows], [int(r["p3"].argmax()) for r in rows]),
        "cls5": eval_classification([r["t5"] for r in rows], [int(r["p5"].argmax()) for r in rows]),
        "progress": eval_regression([r["tp"] for r in rows], [r["pp"] for r in rows]),
        "plateau": eval_classification(
            [r["tpl"] for r in rows], [1 if r["ppl"] >= 0.5 else 0 for r in rows]
        ),
    }


def cell_of(m):
    return {
        "cls3_acc": float(m["cls3"]["acc"]),
        "cls3_f1": float(m["cls3"]["macro_f1"]),
        "cls5_acc": float(m["cls5"]["acc"]),
        "cls5_f1": float(m["cls5"]["macro_f1"]),
        "progress_mae": float(m["progress"]["mae"]),
        "progress_spearman": float(m["progress"]["spearman"]),
        "plateau_acc": float(m["plateau"]["acc"]),
    }


@torch.no_grad()
def eval_lstm_rows(model, loader, device):
    model.eval()
    rows = []
    for x, lens, seg, y3, y5, yp, ypl, g, pid in loader:
        o3, o5, op, opl = model(x.to(device), lens, seg.to(device))
        p3 = F.softmax(o3, dim=-1).cpu().numpy()
        p5 = F.softmax(o5, dim=-1).cpu().numpy()
        for i in range(o3.size(0)):
            rows.append(
                {
                    "key": (int(g[i]), int(pid[i])),
                    "p3": p3[i],
                    "p5": p5[i],
                    "pp": float(op[i].cpu()),
                    "ppl": float(torch.sigmoid(opl[i]).cpu()),
                    "t3": int(y3[i]),
                    "t5": int(y5[i]),
                    "tp": float(yp[i]),
                    "tpl": int(ypl[i]),
                }
            )
    return rows


def ens(rowlists):
    out = []
    for i in range(len(rowlists[0])):
        r = dict(rowlists[0][i])
        r["p3"] = np.mean([rl[i]["p3"] for rl in rowlists], 0)
        r["p5"] = np.mean([rl[i]["p5"] for rl in rowlists], 0)
        r["pp"] = float(np.mean([rl[i]["pp"] for rl in rowlists]))
        r["ppl"] = float(np.mean([rl[i]["ppl"] for rl in rowlists]))
        out.append(r)
    return out


def train_rf_parts(X, meta, fit_parts, pred_parts):
    meta = build_pass_table(X, meta)
    Xp, y3, y5, yp, tags, keys = make_pass_features(X, meta, "nested")
    fit = np.isin(tags, list(fit_parts))
    clf3 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
    clf5 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
    reg = RandomForestRegressor(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
    clf3.fit(Xp[fit], y3[fit])
    clf5.fit(Xp[fit], y5[fit])
    reg.fit(Xp[fit], yp[fit])

    def full(clf, proba, n):
        out = np.zeros((len(proba), n), dtype=np.float64)
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = proba[:, j]
        return out

    packs, metrics = {}, {}
    for part in pred_parts:
        mask = tags == part
        idxs = np.where(mask)[0]
        p3 = full(clf3, clf3.predict_proba(Xp[mask]), 3)
        p5 = full(clf5, clf5.predict_proba(Xp[mask]), 5)
        pp = reg.predict(Xp[mask]).astype(np.float64)
        packs[part] = {
            keys[idx]: {"p3": p3[ii], "p5": p5[ii], "pp": float(pp[ii])}
            for ii, idx in enumerate(idxs)
        }
        metrics[part] = {
            "cls3": eval_classification(y3[mask], clf3.predict(Xp[mask])),
            "cls5": eval_classification(y5[mask], clf5.predict(Xp[mask])),
            "progress": eval_regression(yp[mask], reg.predict(Xp[mask])),
        }
    return packs, metrics


def soft_fuse(e_c3, e_c5, rf, w=SOFT_W):
    rows = []
    for i, r in enumerate(e_c3):
        rf_r = rf[r["key"]]
        rows.append(
            {
                **r,
                "p3": w["p3_lstm"] * e_c3[i]["p3"] + w["p3_rf"] * rf_r["p3"],
                "p5": w["p5_lstm"] * e_c5[i]["p5"] + w["p5_rf"] * rf_r["p5"],
                "ppl": e_c3[i]["ppl"],
            }
        )
    return rows


def apply_prog(base, e_mae, rf, cnn_pp, w):
    out = []
    for i, r in enumerate(base):
        key = r["key"]
        pp = (
            w["resnet"] * float(cnn_pp[key])
            + w["rf"] * float(rf[key]["pp"])
            + w["lstm"] * float(e_mae[i]["pp"])
        )
        nr = dict(r)
        nr["pp"] = float(pp)
        out.append(nr)
    return out


def protocol_rows(e_c3, e_c5, e_mae, rf, cnn_pp, prog_w):
    return apply_prog(soft_fuse(e_c3, e_c5, rf), e_mae, rf, cnn_pp, prog_w)


def train_lstm_select_val(
    seed, X, meta, names, seg, rf_val, device, epochs, max_len, batch_size, lr, out_dir
):
    set_seed(seed)
    device_t = torch.device(device)
    train_ds = PassSeq(X, meta, seg, "train", max_len=max_len)
    val_ds = PassSeq(X, meta, seg, "val", max_len=max_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    model = TemporalExpert(len(names)).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best = {
        "c3": (-1.0, None, None, 0),
        "c5": (-1.0, None, None, 0),
        "mae": (1e9, None, None, 0),
    }

    for ep in range(1, epochs + 1):
        model.train()
        for x, lens, seg_b, y3, y5, yp, ypl, g, pid in train_loader:
            x, seg_b = x.to(device_t), seg_b.to(device_t)
            y3, y5, yp, ypl = y3.to(device_t), y5.to(device_t), yp.to(device_t), ypl.to(device_t)
            o3, o5, op, opl = model(x, lens, seg_b)
            loss = compute_loss(o3, o5, op, opl, y3, y5, yp, ypl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        rows_l = eval_lstm_rows(model, val_loader, device_t)
        # soft-fuse with RF on val for specialty selection (matches main protocol)
        fused_c3 = soft_fuse(rows_l, rows_l, rf_val)
        # for selection scores use fused cls / lstm progress
        m = metrics_from_rows(
            [
                {
                    **fused_c3[i],
                    "pp": 0.25 * rows_l[i]["pp"] + 0.75 * rf_val[rows_l[i]["key"]]["pp"],
                }
                for i in range(len(rows_l))
            ]
        )
        m_l = metrics_from_rows(rows_l)
        print(
            f"[NestedLSTM|s{seed}] ep{ep}/{epochs} val "
            f"fused {m['cls3']['acc']:.4f}/{m['cls5']['acc']:.4f}/{m['progress']['mae']:.4f} "
            f"lstm3={m_l['cls3']['acc']:.4f}",
            flush=True,
        )
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if m["cls3"]["acc"] > best["c3"][0]:
            best["c3"] = (m["cls3"]["acc"], state, rows_l, ep)
        if m["cls5"]["acc"] > best["c5"][0]:
            best["c5"] = (m["cls5"]["acc"], state, rows_l, ep)
        if m["progress"]["mae"] < best["mae"][0]:
            best["mae"] = (m["progress"]["mae"], state, rows_l, ep)

    ckpt = {
        "st_c3": best["c3"][1],
        "st_c5": best["c5"][1],
        "st_mae": best["mae"][1],
        "val_ep": {"c3": best["c3"][3], "c5": best["c5"][3], "mae": best["mae"][3]},
        "val_scores": {
            "c3": best["c3"][0],
            "c5": best["c5"][0],
            "mae": best["mae"][0],
        },
        "seed": seed,
        "selection": "nested_val",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ours_nested_s{seed}.pt"
    torch.save(ckpt, path)
    print(f"  saved {path} val_ep={ckpt['val_ep']}", flush=True)
    return ckpt, path


def refit_lstm_fixed_epochs(
    seed, X, meta, names, seg, device, epochs, max_len, batch_size, lr, out_dir, tag="refit"
):
    """Train on train∪val for a fixed epoch budget (no test peeking)."""
    set_seed(seed)
    device_t = torch.device(device)
    train_ds = PassSeq(X, meta, seg, "train", max_len=max_len, fit_parts=("train", "val"))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    model = TemporalExpert(len(names)).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(1, epochs + 1):
        model.train()
        for x, lens, seg_b, y3, y5, yp, ypl, g, pid in train_loader:
            x, seg_b = x.to(device_t), seg_b.to(device_t)
            y3, y5, yp, ypl = y3.to(device_t), y5.to(device_t), yp.to(device_t), ypl.to(device_t)
            o3, o5, op, opl = model(x, lens, seg_b)
            loss = compute_loss(o3, o5, op, opl, y3, y5, yp, ypl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        if ep % 10 == 0 or ep == epochs:
            print(f"[NestedLSTM-refit|s{seed}] ep{ep}/{epochs}", flush=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # same weights for all specialty slots (epoch budget fixed from val selection)
    ckpt = {
        "st_c3": state,
        "st_c5": state,
        "st_mae": state,
        "seed": seed,
        "refit_epochs": epochs,
        "selection": "nested_val_then_refit_trainval",
    }
    path = out_dir / f"ours_nested_{tag}_s{seed}.pt"
    torch.save(ckpt, path)
    return ckpt, path


def eval_specialists(X, meta, names, seg, part, device, ckpt_path, max_len=24):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ds = PassSeq(X, meta, seg, part, max_len=max_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
    buckets = {}
    for key in ("st_c3", "st_c5", "st_mae"):
        model = TemporalExpert(len(names)).to(device)
        model.load_state_dict(state[key])
        buckets[key] = eval_lstm_rows(model, loader, device)
    return buckets


def train_resnet_select_val(frames_meta, device, epochs=16, batch_size=64):
    device_t = torch.device(device)
    train_ds = FrameDataset(frames_meta, "nested", "train", train=True)
    val_ds = FrameDataset(frames_meta, "nested", "val", train=False)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    model = MultiTaskCNN().to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    best = None
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y3, y5, yp in train_loader:
            x = x.to(device_t)
            y3, y5, yp = y3.to(device_t), y5.to(device_t), yp.to(device_t)
            o3, o5, op = model(x)
            loss = ce(o3, y3) + ce(o5, y5) + 2.0 * mse(op, yp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        pp, tp = [], []
        with torch.no_grad():
            for x, y3, y5, yp in val_loader:
                _, _, op = model(x.to(device_t))
                pp.append(op.cpu().numpy())
                tp.append(yp.numpy())
        m = eval_regression(np.concatenate(tp), np.concatenate(pp))
        score = (m["spearman"], -m["mae"])
        print(
            f"[NestedResNet] ep{ep}/{epochs} val mae={m['mae']:.4f} sp={m['spearman']:.6f}",
            flush=True,
        )
        if best is None or score > best[0]:
            best = (score, m, ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = OUT / "resnet18_nested_best_spearman.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "metrics": best[1], "epoch": best[2], "selection": "val"}, out)
    return out, best[2], best[1]


def refit_resnet_fixed(frames_meta, device, epochs, batch_size=64):
    device_t = torch.device(device)
    # train∪val
    meta = frames_meta.copy()
    if "split_nested" not in meta.columns:
        meta = annotate_nested_split(meta)
    keep = meta["split_nested"].isin(["train", "val"])
    # temporary: mark both as train for FrameDataset interleaved-style reuse
    meta2 = meta.copy()
    meta2["split_nested"] = np.where(keep, "train", meta2["split_nested"])
    train_ds = FrameDataset(meta2, "nested", "train", train=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    model = MultiTaskCNN().to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y3, y5, yp in train_loader:
            x = x.to(device_t)
            y3, y5, yp = y3.to(device_t), y5.to(device_t), yp.to(device_t)
            o3, o5, op = model(x)
            loss = ce(o3, y3) + ce(o5, y5) + 2.0 * mse(op, yp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        if ep % 4 == 0 or ep == epochs:
            print(f"[NestedResNet-refit] ep{ep}/{epochs}", flush=True)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = OUT / "resnet18_nested_refit.pt"
    torch.save({"model": state, "refit_epochs": epochs}, out)
    return out


@torch.no_grad()
def cnn_pass_progress(ckpt, frames_meta, part, device, max_per_pass=4):
    device_t = torch.device(device)
    model = MultiTaskCNN().to(device_t)
    blob = torch.load(ckpt, map_location=device_t, weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    meta = annotate_nested_split(frames_meta) if "split_nested" not in frames_meta.columns else frames_meta
    meta = meta[meta["split_nested"] == part].reset_index(drop=True)
    tf = FrameDataset(meta, "nested", part, train=False).tf
    out = {}
    for (grind, pass_id), gdf in meta.groupby(["grind", "pass_id"], sort=True):
        gdf = gdf.sort_values("frame_in_pass")
        idx = np.linspace(0, len(gdf) - 1, min(max_per_pass, len(gdf)), dtype=int)
        preds = []
        for i in idx:
            img = __import__("PIL").Image.open(gdf.iloc[i]["path"]).convert("RGB")
            x = tf(img).unsqueeze(0).to(device_t)
            _, _, op = model(x)
            preds.append(float(op.cpu()))
        out[(int(grind), int(pass_id))] = float(np.mean(preds))
    return out


@torch.no_grad()
def cnn_frame_progress(ckpt, frames_meta, part, device):
    device_t = torch.device(device)
    model = MultiTaskCNN().to(device_t)
    blob = torch.load(ckpt, map_location=device_t, weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    ds = FrameDataset(annotate_nested_split(frames_meta), "nested", part, train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    pp, tp = [], []
    with torch.no_grad():
        for x, y3, y5, yp in loader:
            _, _, op = model(x.to(device_t))
            pp.append(op.cpu().numpy())
            tp.append(yp.numpy())
    return np.concatenate(tp), np.concatenate(pp)


def search_prog_w(e_c3, e_c5, e_mae, rf, cnn_pp, floor_c3, floor_c5):
    """Convex grid on val; minimize MAE with Acc floors."""
    best = None
    # denser grid near prior main weights
    for wr in np.linspace(0.40, 0.85, 10):
        for wf in np.linspace(0.05, 0.50, 10):
            wl = 1.0 - wr - wf
            if wl < -1e-9 or wl > 0.40:
                continue
            if wl < 0:
                continue
            w = {"resnet": float(wr), "rf": float(wf), "lstm": float(wl)}
            rows = protocol_rows(e_c3, e_c5, e_mae, rf, cnn_pp, w)
            m = metrics_from_rows(rows)
            if m["cls3"]["acc"] + 1e-12 < floor_c3:
                continue
            if m["cls5"]["acc"] + 1e-12 < floor_c5:
                continue
            score = (m["progress"]["mae"], -m["progress"]["spearman"])
            if best is None or score < best[0]:
                best = (score, w, cell_of(m))
    if best is None:
        # fallback: ignore floors, minimize MAE
        for wr in np.linspace(0.40, 0.85, 10):
            for wf in np.linspace(0.05, 0.50, 10):
                wl = 1.0 - wr - wf
                if wl < 0 or wl > 0.40:
                    continue
                w = {"resnet": float(wr), "rf": float(wf), "lstm": float(wl)}
                rows = protocol_rows(e_c3, e_c5, e_mae, rf, cnn_pp, w)
                m = metrics_from_rows(rows)
                score = (m["progress"]["mae"], -m["progress"]["spearman"])
                if best is None or score < best[0]:
                    best = (score, w, cell_of(m))
    return best[1], best[2]


def train_b6_lstm(X, meta, names, seg, device, epochs, max_len, batch_size, lr):
    """Baseline BiLSTM: train on train, select Acc3 on val, eval on test."""
    set_seed(0)
    device_t = torch.device(device)
    train_ds = PassSeq(X, meta, seg, "train", max_len=max_len)
    val_ds = PassSeq(X, meta, seg, "val", max_len=max_len)
    test_ds = PassSeq(X, meta, seg, "test", max_len=max_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    model = TemporalExpert(len(names)).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, lens, seg_b, y3, y5, yp, ypl, g, pid in train_loader:
            x, seg_b = x.to(device_t), seg_b.to(device_t)
            y3, y5, yp, ypl = y3.to(device_t), y5.to(device_t), yp.to(device_t), ypl.to(device_t)
            o3, o5, op, opl = model(x, lens, seg_b)
            loss = compute_loss(o3, o5, op, opl, y3, y5, yp, ypl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        rows = eval_lstm_rows(model, val_loader, device_t)
        m = metrics_from_rows(rows)
        if m["cls3"]["acc"] > best_acc:
            best_acc = m["cls3"]["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"[B6-nested] ep{ep} val Acc3={m['cls3']['acc']:.4f}", flush=True)
    model.load_state_dict(best_state)
    # refit train∪val fixed epochs = selection ep approx: use full epochs on train∪val
    # simpler: evaluate selected ckpt trained only on train (honest); also refit
    test_rows = eval_lstm_rows(model, test_loader, device_t)
    return cell_of(metrics_from_rows(test_rows)), best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device2", default="cuda:1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--resnet-epochs", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--phase", default="all", choices=["all", "select", "refit", "eval"])
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, meta, names = load_feature_cache(ROOT / "outputs/baselines/features")
    meta = annotate_nested_split(meta)
    frames_meta = annotate_nested_split(
        load_frames_meta(ROOT / "outputs/baselines/features/frame_features_meta.csv")
    )
    seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    print("=== Nested split counts ===", flush=True)
    print(meta.groupby("split_nested").size(), flush=True)
    print("stages:", meta.groupby("split_nested")["grind"].nunique().to_dict(), flush=True)

    # ---- RF baselines on nested ----
    print("=== RF fit train → val/test ===", flush=True)
    rf_packs, rf_metrics = train_rf_parts(X, meta, ("train",), ("val", "test"))
    print("B5 val:", {k: rf_metrics["val"][k] if k != "progress" else rf_metrics["val"][k] for k in rf_metrics["val"]}, flush=True)
    print(
        f"B5 nested train→test: Acc3={rf_metrics['test']['cls3']['acc']:.4f} "
        f"Acc5={rf_metrics['test']['cls5']['acc']:.4f} MAE={rf_metrics['test']['progress']['mae']:.4f}",
        flush=True,
    )

    # ---- Phase: LSTM selection on val ----
    sel_dir = OUT / "select"
    if not args.skip_train and args.phase in ("all", "select"):
        print("=== LSTM specialty selection on val ===", flush=True)
        # run seeds sequentially on primary device (stable); user has time
        for seed in seeds:
            train_lstm_select_val(
                seed,
                X,
                meta,
                names,
                seg,
                rf_packs["val"],
                args.device,
                args.epochs,
                args.max_len,
                args.batch_size,
                args.lr,
                sel_dir,
            )

    # ---- ResNet selection on val ----
    resnet_sel = OUT / "resnet18_nested_best_spearman.pt"
    resnet_ep = 12
    if not args.skip_train and args.phase in ("all", "select"):
        print("=== ResNet selection on val ===", flush=True)
        resnet_sel, resnet_ep, _ = train_resnet_select_val(
            frames_meta, args.device, epochs=args.resnet_epochs, batch_size=args.batch_size
        )
    elif resnet_sel.exists():
        blob = torch.load(resnet_sel, map_location="cpu", weights_only=False)
        resnet_ep = int(blob.get("epoch", 12))

    # Collect val specialists for weight search
    print("=== Progress-weight search on val ===", flush=True)
    device_t = torch.device(args.device)
    all_c3, all_c5, all_mae = [], [], []
    for seed in seeds:
        ck = sel_dir / f"ours_nested_s{seed}.pt"
        b = eval_specialists(X, meta, names, seg, "val", device_t, ck, args.max_len)
        all_c3.append(b["st_c3"])
        all_c5.append(b["st_c5"])
        all_mae.append(b["st_mae"])
    e_c3, e_c5, e_mae = ens(all_c3), ens(all_c5), ens(all_mae)
    cnn_pp_val = cnn_pass_progress(resnet_sel, frames_meta, "val", args.device)
    floor_c3 = float(rf_metrics["val"]["cls3"]["acc"])  # at least match B5 on val; B6 below
    # also require not worse than B5 cls5
    floor_c5 = float(rf_metrics["val"]["cls5"]["acc"])
    # B6 val Acc3 as additional floor if available later; use max(B5, 0.95*prior)
    prog_w, val_cell = search_prog_w(
        e_c3, e_c5, e_mae, rf_packs["val"], cnn_pp_val, floor_c3=min(floor_c3, 0.93), floor_c5=min(floor_c5, 0.90)
    )
    # Prefer floors matching B5 val when possible; retry with true floors
    prog_w2, val_cell2 = search_prog_w(
        e_c3, e_c5, e_mae, rf_packs["val"], cnn_pp_val, floor_c3=floor_c3, floor_c5=floor_c5
    )
    if val_cell2 is not None:
        prog_w, val_cell = prog_w2, val_cell2
    print(f"Selected prog_w on val: {prog_w}", flush=True)
    print(f"Val ensemble cell: {val_cell}", flush=True)
    save_json({"prog_w": prog_w, "val_cell": val_cell, "soft_w": SOFT_W}, OUT / "selected_on_val.json")

    # ---- Refit RF/ResNet on train∪val; LSTM keeps val-selected specialty heads ----
    refit_dir = OUT / "refit"
    refit_dir.mkdir(parents=True, exist_ok=True)
    ep_budgets = []
    for seed in seeds:
        ck = torch.load(sel_dir / f"ours_nested_s{seed}.pt", map_location="cpu", weights_only=False)
        ep_budgets.append(max(ck["val_ep"].values()))
    refit_epochs = int(np.clip(np.mean(ep_budgets), 40, args.epochs))
    print(f"=== Refit ResNet train∪val epochs={max(resnet_ep, 12)} (LSTM specialty kept) ===", flush=True)
    if not args.skip_train and args.phase in ("all", "refit"):
        resnet_refit = refit_resnet_fixed(
            frames_meta, args.device, epochs=max(resnet_ep, 12), batch_size=args.batch_size
        )
    else:
        resnet_refit = OUT / "resnet18_nested_refit.pt"
        if not resnet_refit.exists():
            resnet_refit = resnet_sel

    # RF refit train∪val
    print("=== RF refit train∪val → test ===", flush=True)
    rf_refit, rf_refit_m = train_rf_parts(X, meta, ("train", "val"), ("test", "val"))
    b5_test = cell_of(
        {
            "cls3": rf_refit_m["test"]["cls3"],
            "cls5": rf_refit_m["test"]["cls5"],
            "progress": rf_refit_m["test"]["progress"],
            "plateau": {"acc": float("nan"), "macro_f1": float("nan")},
        }
    )
    # plateau dummy for cell_of compatibility — rebuild manually
    b5_test = {
        "cls3_acc": float(rf_refit_m["test"]["cls3"]["acc"]),
        "cls3_f1": float(rf_refit_m["test"]["cls3"]["macro_f1"]),
        "cls5_acc": float(rf_refit_m["test"]["cls5"]["acc"]),
        "cls5_f1": float(rf_refit_m["test"]["cls5"]["macro_f1"]),
        "progress_mae": float(rf_refit_m["test"]["progress"]["mae"]),
        "progress_spearman": float(rf_refit_m["test"]["progress"]["spearman"]),
    }
    print(f"B5 refit→test: {b5_test}", flush=True)

    # B6: select on val then refit
    print("=== B6 nested select+refit ===", flush=True)
    b6_cell, b6_val = train_b6_lstm(
        X, meta, names, seg, args.device, args.epochs, args.max_len, args.batch_size, args.lr
    )
    # Better B6: refit train∪val
    set_seed(0)
    device_t = torch.device(args.device)
    train_ds = PassSeq(X, meta, seg, "train", max_len=args.max_len, fit_parts=("train", "val"))
    test_ds = PassSeq(X, meta, seg, "test", max_len=args.max_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    model = TemporalExpert(len(names)).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=refit_epochs)
    for ep in range(1, refit_epochs + 1):
        model.train()
        for x, lens, seg_b, y3, y5, yp, ypl, g, pid in train_loader:
            x, seg_b = x.to(device_t), seg_b.to(device_t)
            y3, y5, yp, ypl = y3.to(device_t), y5.to(device_t), yp.to(device_t), ypl.to(device_t)
            o3, o5, op, opl = model(x, lens, seg_b)
            loss = compute_loss(o3, o5, op, opl, y3, y5, yp, ypl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    b6_cell = cell_of(metrics_from_rows(eval_lstm_rows(model, test_loader, device_t)))
    print(f"B6 refit→test: {b6_cell}", flush=True)

    # B3 ResNet: use refit ckpt frame-level / pass-level as baseline on test
    print("=== B3 nested ResNet test ===", flush=True)
    tp, pp = cnn_frame_progress(resnet_refit, frames_meta, "test", args.device)
    # classification from same multitask model at frame level then majority? Paper B3 is frame-level.
    # Re-eval full multitask on test frames:
    device_t = torch.device(args.device)
    model = MultiTaskCNN().to(device_t)
    model.load_state_dict(torch.load(resnet_refit, map_location=device_t, weights_only=False)["model"])
    model.eval()
    ds = FrameDataset(frames_meta, "nested", "test", train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    p3, t3, p5, t5, pprog, tprog = [], [], [], [], [], []
    with torch.no_grad():
        for x, y3, y5, yp in loader:
            o3, o5, op = model(x.to(device_t))
            p3.append(o3.argmax(1).cpu().numpy())
            t3.append(y3.numpy())
            p5.append(o5.argmax(1).cpu().numpy())
            t5.append(y5.numpy())
            pprog.append(op.cpu().numpy())
            tprog.append(yp.numpy())
    b3_cell = {
        "cls3_acc": float(eval_classification(np.concatenate(t3), np.concatenate(p3))["acc"]),
        "cls3_f1": float(eval_classification(np.concatenate(t3), np.concatenate(p3))["macro_f1"]),
        "cls5_acc": float(eval_classification(np.concatenate(t5), np.concatenate(p5))["acc"]),
        "cls5_f1": float(eval_classification(np.concatenate(t5), np.concatenate(p5))["macro_f1"]),
        "progress_mae": float(eval_regression(np.concatenate(tprog), np.concatenate(pprog))["mae"]),
        "progress_spearman": float(
            eval_regression(np.concatenate(tprog), np.concatenate(pprog))["spearman"]
        ),
    }
    print(f"B3 refit→test: {b3_cell}", flush=True)

    # ---- Ours final eval on test (once) ----
    # Specialty LSTMs: train-only + val-selected heads (keeps task specialty).
    # RF/ResNet: refit on train∪val with frozen epoch/weight choices.
    print("=== PASS-Ord nested final test (once) ===", flush=True)
    all_c3, all_c5, all_mae = [], [], []
    per_seed = []
    cnn_pp_te = cnn_pass_progress(resnet_refit, frames_meta, "test", args.device)
    for seed in seeds:
        ck = sel_dir / f"ours_nested_s{seed}.pt"
        b = eval_specialists(X, meta, names, seg, "test", device_t, ck, args.max_len)
        rows = protocol_rows(b["st_c3"], b["st_c5"], b["st_mae"], rf_refit["test"], cnn_pp_te, prog_w)
        c = cell_of(metrics_from_rows(rows))
        c["seed"] = seed
        per_seed.append(c)
        all_c3.append(b["st_c3"])
        all_c5.append(b["st_c5"])
        all_mae.append(b["st_mae"])
        print(
            f"  seed{seed}: Acc3={c['cls3_acc']:.4f} Acc5={c['cls5_acc']:.4f} "
            f"MAE={c['progress_mae']:.4f} sp={c['progress_spearman']:.4f}",
            flush=True,
        )
    e_c3, e_c5, e_mae = ens(all_c3), ens(all_c5), ens(all_mae)
    ens_rows = protocol_rows(e_c3, e_c5, e_mae, rf_refit["test"], cnn_pp_te, prog_w)
    ens_cell = cell_of(metrics_from_rows(ens_rows))

    # Frame-level progress for main table MAE/Spearman
    # Broadcast RF/LSTM pass progress to frames in test passes
    meta_p = build_pass_table(X, meta)
    frame_tp, frame_pp = [], []
    # map pass key -> fused progress from ens_rows
    pass_pp = {r["key"]: r["pp"] for r in ens_rows}
    te_frames = frames_meta[frames_meta["split_nested"] == "test"]
    # ResNet frame preds
    tp_f, pp_f = cnn_frame_progress(resnet_refit, frames_meta, "test", args.device)
    # For each test frame, blend using same prog_w with pass-level RF/LSTM
    # Build pass progress maps
    rf_pp = {k: v["pp"] for k, v in rf_refit["test"].items()}
    lstm_pp = {r["key"]: r["pp"] for r in e_mae}
    # align frame order with FrameDataset
    ds_te = FrameDataset(frames_meta, "nested", "test", train=False)
    meta_te = ds_te.meta.reset_index(drop=True)
    assert len(meta_te) == len(pp_f)
    fused_frame = []
    frame_tp = []
    for i in range(len(meta_te)):
        r = meta_te.iloc[i]
        key = (int(r.grind), int(r.pass_id))
        pp = (
            prog_w["resnet"] * float(pp_f[i])
            + prog_w["rf"] * float(rf_pp.get(key, pp_f[i]))
            + prog_w["lstm"] * float(lstm_pp.get(key, pp_f[i]))
        )
        fused_frame.append(pp)
        frame_tp.append(progress(int(r.grind)))
    frame_prog = eval_regression(np.asarray(frame_tp), np.asarray(fused_frame))
    ens_cell_frame = dict(ens_cell)
    ens_cell_frame["progress_mae"] = float(frame_prog["mae"])
    ens_cell_frame["progress_spearman"] = float(frame_prog["spearman"])
    print(f"Ensemble pass-level: {ens_cell}", flush=True)
    print(f"Ensemble frame progress: {ens_cell_frame}", flush=True)

    # SOTA check vs nested baselines
    best_c3 = max(b5_test["cls3_acc"], b6_cell["cls3_acc"], b3_cell["cls3_acc"])
    best_c5 = max(b5_test["cls5_acc"], b6_cell["cls5_acc"], b3_cell["cls5_acc"])
    best_mae = min(b5_test["progress_mae"], b6_cell["progress_mae"], b3_cell["progress_mae"])
    best_sp = max(
        b5_test["progress_spearman"], b6_cell["progress_spearman"], b3_cell["progress_spearman"]
    )
    sota = {
        "cls3": ens_cell["cls3_acc"] + 1e-12 >= best_c3,
        "cls5": ens_cell["cls5_acc"] + 1e-12 >= best_c5,
        "mae": ens_cell_frame["progress_mae"] - 1e-12 <= best_mae,
        "spearman": ens_cell_frame["progress_spearman"] + 1e-12 >= best_sp,
    }
    print(f"SOTA flags: {sota}", flush=True)
    print(
        f"Baselines best: Acc3={best_c3:.4f} Acc5={best_c5:.4f} MAE={best_mae:.4f} sp={best_sp:.4f}",
        flush=True,
    )

    summary = {
        "protocol": "nested_interleaved_select_val_refit_trainval_eval_test_once",
        "prog_w": prog_w,
        "soft_w": SOFT_W,
        "refit_epochs_lstm": refit_epochs,
        "resnet_select_ep": resnet_ep,
        "baselines": {"B5": b5_test, "B6": b6_cell, "B3": b3_cell},
        "ours_per_seed": per_seed,
        "ours_ens_pass": ens_cell,
        "ours_ens_frame_prog": ens_cell_frame,
        "sota": sota,
        "elapsed_s": time.time() - t0,
    }
    save_json(summary, OUT / "nested_sota_summary.json")
    rows_out = [
        {"model": "B3_ResNet18", **b3_cell},
        {"model": "B5_MeanPool_RF", **b5_test},
        {"model": "B6_LSTM", **{k: b6_cell[k] for k in b5_test}},
        {"model": "Ours_PASSOrd_nested", **{k: ens_cell_frame[k] for k in b5_test}},
    ]
    # fill f1 for ours from ens_cell
    rows_out[-1]["cls3_f1"] = ens_cell["cls3_f1"]
    rows_out[-1]["cls5_f1"] = ens_cell["cls5_f1"]
    pd.DataFrame(rows_out).to_csv(TABLES / "T4_nested_main.csv", index=False)
    pd.DataFrame(per_seed).to_csv(TABLES / "nested_per_seed.csv", index=False)
    print(f"Wrote {OUT / 'nested_sota_summary.json'} and T4_nested_main.csv", flush=True)
    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
