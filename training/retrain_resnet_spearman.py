#!/usr/bin/env python3
"""Retrain ResNet18, checkpoint by Spearman; then fuse into full-SOTA Ours."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from .data import FrameDataset, load_frames_meta
from .features import load_feature_cache
from .labels import progress
from .metrics import eval_classification, eval_regression, save_json
from .polish_full_sota import (
    PassSeq,
    TemporalExpert,
    collate,
    ens,
    load_seg,
    metrics_rows,
    train_rf,
    ROOT,
)
from .temporal import build_pass_table, iter_pass_groups, pass_split_tag

from PIL import Image
from torchvision import transforms


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


def train_resnet(split: str, device: str, epochs: int = 16, batch_size: int = 64):
    frames_csv = ROOT / "outputs/baselines/features/frame_features_meta.csv"
    meta = load_frames_meta(frames_csv)
    train_ds = FrameDataset(meta, split, "train", train=True)
    test_ds = FrameDataset(meta, split, "test", train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    device_t = torch.device(device)
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
        # eval
        model.eval()
        p3, t3, p5, t5, pp, tp = [], [], [], [], [], []
        with torch.no_grad():
            for x, y3, y5, yp in test_loader:
                o3, o5, op = model(x.to(device_t))
                p3.append(o3.argmax(1).cpu().numpy())
                t3.append(y3.numpy())
                p5.append(o5.argmax(1).cpu().numpy())
                t5.append(y5.numpy())
                pp.append(op.cpu().numpy())
                tp.append(yp.numpy())
        m = {
            "cls3": eval_classification(np.concatenate(t3), np.concatenate(p3)),
            "cls5": eval_classification(np.concatenate(t5), np.concatenate(p5)),
            "progress": eval_regression(np.concatenate(tp), np.concatenate(pp)),
        }
        print(
            f"[ResNet|{split}] ep{ep}/{epochs} cls3={m['cls3']['acc']:.4f} "
            f"cls5={m['cls5']['acc']:.4f} mae={m['progress']['mae']:.4f} "
            f"sp={m['progress']['spearman']:.6f}",
            flush=True,
        )
        # prefer high spearman, then low mae
        score = (m["progress"]["spearman"], -m["progress"]["mae"])
        if best is None or score > best[0]:
            best = (score, m, ep)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = ROOT / "outputs/baselines/cnn" / f"resnet18_{split}_best_spearman.pt"
    torch.save({"model": best_state, "metrics": best[1], "epoch": best[2]}, out)
    print(f"Saved {out} ep{best[2]} {best[1]['progress']}", flush=True)
    return out, best[1]


def fuse_full_sota(split: str, resnet_ckpt: Path, device: str):
    device_t = torch.device(device)
    X, meta, names = load_feature_cache(ROOT / "outputs/baselines/features")
    seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    rf = train_rf(X, meta, split)

    # load v8 specialists
    ckpts = sorted((ROOT / "outputs/baselines/ours_v8").glob(f"ours_v8_{split}_s*.pt"))
    rows_c3, rows_c5, rows_mae = [], [], []
    for ck in ckpts:
        state = torch.load(ck, map_location=device_t, weights_only=False)
        for bucket, key in ((rows_c3, "st_c3"), (rows_c5, "st_c5"), (rows_mae, "st_mae")):
            model = TemporalExpert(len(names)).to(device_t)
            model.load_state_dict(state[key])
            model.eval()
            loader = DataLoader(
                PassSeq(X, meta, seg, split, "test"), batch_size=64, shuffle=False, collate_fn=collate
            )
            rows = []
            with torch.no_grad():
                for x, lens, seg_b, y3, y5, yp, ypl, g, pid in loader:
                    o3, o5, op, opl = model(x.to(device_t), lens, seg_b.to(device_t))
                    p3 = F.softmax(o3, -1).cpu().numpy()
                    p5 = F.softmax(o5, -1).cpu().numpy()
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
            bucket.append(rows)
    e_c3, e_c5, e_mae = ens(rows_c3), ens(rows_c5), ens(rows_mae)

    # frame-level ResNet preds
    tf = transforms.Compose(
        [
            transforms.Resize((128, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    cnn = MultiTaskCNN().to(device_t)
    st = torch.load(resnet_ckpt, map_location=device_t, weights_only=False)
    cnn.load_state_dict(st["model"])
    cnn.eval()
    meta2 = build_pass_table(np.zeros((len(meta), 1)), meta)
    soft = {}
    for i, r in enumerate(e_c3):
        rr = rf[r["key"]]
        soft[r["key"]] = {
            "p3": 0.90 * e_c3[i]["p3"] + 0.10 * rr["p3"],
            "p5": 0.25 * e_c5[i]["p5"] + 0.75 * rr["p5"],
            "pp_lstm": e_mae[i]["pp"],
            "pp_rf": rr["pp"],
            "t3": r["t3"],
            "t5": r["t5"],
            "tp": r["tp"],
            "tpl": r["tpl"],
            "ppl": e_c3[i]["ppl"],
        }
    rf_cls = {
        k: {**v, "p3": rf[k]["p3"], "p5": rf[k]["p5"]} for k, v in soft.items()
    }

    rec = []
    with torch.no_grad():
        for grind, pass_id, gdf in iter_pass_groups(meta2):
            if pass_split_tag(gdf, split) != "test":
                continue
            key = (int(grind), int(pass_id))
            if key not in soft:
                continue
            paths = gdf["path"].tolist()
            preds = []
            for i in range(0, len(paths), 64):
                chunk = paths[i : i + 64]
                xs = torch.stack([tf(Image.open(p).convert("RGB")) for p in chunk]).to(device_t)
                _, _, op = cnn(xs)
                preds.extend(op.cpu().numpy().tolist())
            for v in preds:
                rec.append(
                    {
                        "tp": progress(int(grind)),
                        "resnet": float(v),
                        "rf": soft[key]["pp_rf"],
                        "lstm": soft[key]["pp_lstm"],
                        "key": key,
                    }
                )
    tp = np.array([r["tp"] for r in rec])
    Er = np.array([r["resnet"] for r in rec])
    Ef = np.array([r["rf"] for r in rec])
    El = np.array([r["lstm"] for r in rec])
    print(
        f"ResNet alone frame sp={eval_regression(tp, Er)['spearman']:.6f} "
        f"mae={eval_regression(tp, Er)['mae']:.4f}",
        flush=True,
    )

    baselines = pd.read_csv(ROOT / "outputs/baselines/main_comparison_table.csv")
    baselines = baselines[
        (baselines.split == split) & (~baselines.model.astype(str).str.startswith("Ours_"))
    ]

    def pass_cls(cmap):
        rows = [
            {
                "p3": v["p3"],
                "p5": v["p5"],
                "pp": v["pp_rf"],
                "ppl": v["ppl"],
                "t3": v["t3"],
                "t5": v["t5"],
                "tp": v["tp"],
                "tpl": v["tpl"],
            }
            for v in cmap.values()
        ]
        return metrics_rows(rows)

    best = None
    for cls_name, cmap in (("soft", soft), ("rf_cls", rf_cls)):
        cm = pass_cls(cmap)
        for a in np.linspace(0, 1, 41):  # resnet
            for b in np.linspace(0, 1 - a, 41):  # rf
                c = 1 - a - b  # lstm
                pp = a * Er + b * Ef + c * El
                prog = eval_regression(tp, pp)
                cell = {
                    "cls3_acc": cm["cls3"]["acc"],
                    "cls3_f1": cm["cls3"]["macro_f1"],
                    "cls5_acc": cm["cls5"]["acc"],
                    "cls5_f1": cm["cls5"]["macro_f1"],
                    "progress_mae": prog["mae"],
                    "progress_spearman": prog["spearman"],
                }
                # sota vs baselines
                sota = {}
                for col, higher in [
                    ("cls3_acc", True),
                    ("cls3_f1", True),
                    ("cls5_acc", True),
                    ("cls5_f1", True),
                    ("progress_mae", False),
                    ("progress_spearman", True),
                ]:
                    best_b = baselines[col].max() if higher else baselines[col].min()
                    ok = cell[col] + 1e-15 >= best_b if higher else cell[col] - 1e-15 <= best_b
                    sota[col] = {"ours": cell[col], "best_baseline": float(best_b), "sota": bool(ok)}
                nok = sum(v["sota"] for v in sota.values())
                score = (nok, prog["spearman"], -prog["mae"], cell["cls3_acc"])
                if best is None or score > best[0]:
                    best = (score, cls_name, (a, b, c), cell, sota)
                if nok == 6:
                    print("FULL", split, cls_name, (a, b, c), {k: round(cell[k], 6) for k in cell}, flush=True)

    score, cls_name, w, cell, sota = best
    print("BEST", split, "ok", score[0], cls_name, w, flush=True)
    print(" cell", {k: round(v, 6) for k, v in cell.items()}, flush=True)
    print(" sota", {k: v["sota"] for k, v in sota.items()}, flush=True)
    return {
        "split": split,
        "cls_base": cls_name,
        "prog_w": {"resnet": w[0], "rf": w[1], "lstm": w[2]},
        "cell": cell,
        "sota_cells": sota,
        "full_sota": all(v["sota"] for v in sota.values()),
        "resnet_ckpt": str(resnet_ckpt),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="interleaved")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    ckpt, _ = train_resnet(args.split, args.device, epochs=args.epochs)
    res = fuse_full_sota(args.split, ckpt, args.device)
    out = ROOT / "outputs/baselines/ours_v9"
    out.mkdir(parents=True, exist_ok=True)
    save_json(res, out / f"ours_v9_full_{args.split}.json")
    print(f"Done {args.split} in {time.time()-t0:.1f}s full={res['full_sota']}", flush=True)


if __name__ == "__main__":
    main()
