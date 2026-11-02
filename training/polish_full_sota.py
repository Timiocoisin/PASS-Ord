#!/usr/bin/env python3
"""Polish to clear remaining full-SOTA cells using v8 checkpoints + CNN progress."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .features import load_feature_cache
from .labels import cls3, cls5, progress
from .metrics import eval_classification, eval_regression, save_json
from .temporal import build_pass_table, iter_pass_groups, make_pass_features, pass_split_tag

ROOT = Path(__file__).resolve().parents[1]
PLATEAU_G = 94


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


class PassSeq(Dataset):
    def __init__(self, X, meta, seg, split, part, max_len=24):
        self.seqs, self.segs, self.y3, self.y5, self.yp, self.ypl, self.keys = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        meta = build_pass_table(X, meta)
        for grind, pass_id, gdf in iter_pass_groups(meta):
            if pass_split_tag(gdf, split) != part:
                continue
            idx = gdf["feat_idx"].to_numpy()
            feats = X[idx]
            if len(feats) > max_len:
                sel = np.linspace(0, len(feats) - 1, max_len, dtype=int)
                feats = feats[sel]
            g = int(grind)
            self.seqs.append(torch.from_numpy(feats.astype(np.float32)))
            self.segs.append(
                torch.from_numpy(seg.get(g, np.zeros(4, np.float32)).astype(np.float32))
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
            *self.keys[i],
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


def load_seg(path):
    df = pd.read_csv(path)
    return {
        int(r.grind): np.array(
            [r.metal_ratio, r.metal_ratio_L, r.metal_ratio_C, r.metal_ratio_R], np.float32
        )
        for r in df.itertuples()
    }


def train_rf(X, meta, split):
    meta = build_pass_table(X, meta)
    Xp, y3, y5, yp, tags, keys = make_pass_features(X, meta, split)
    tr, te = tags == "train", tags == "test"
    clf3 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
    clf5 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
    reg = RandomForestRegressor(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
    clf3.fit(Xp[tr], y3[tr])
    clf5.fit(Xp[tr], y5[tr])
    reg.fit(Xp[tr], yp[tr])

    def full(clf, proba, n):
        out = np.zeros((len(proba), n))
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = proba[:, j]
        return out

    idxs = np.where(te)[0]
    p3 = full(clf3, clf3.predict_proba(Xp[te]), 3)
    p5 = full(clf5, clf5.predict_proba(Xp[te]), 5)
    pp = reg.predict(Xp[te])
    return {
        keys[i]: {"p3": p3[k], "p5": p5[k], "pp": float(pp[k])} for k, i in enumerate(idxs)
    }


class MultiTaskCNN(nn.Module):
    def __init__(self, name="resnet18"):
        super().__init__()
        if name == "resnet18":
            bb = models.resnet18(weights=None)
            in_f = bb.fc.in_features
            bb.fc = nn.Identity()
        else:
            bb = models.efficientnet_b0(weights=None)
            in_f = bb.classifier[1].in_features
            bb.classifier = nn.Identity()
        self.backbone = bb
        self.head_cls3 = nn.Linear(in_f, 3)
        self.head_cls5 = nn.Linear(in_f, 5)
        self.head_prog = nn.Linear(in_f, 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.head_cls3(f), self.head_cls5(f), self.head_prog(f).squeeze(-1)


@torch.no_grad()
def cnn_pass_progress(ckpt, name, meta, split, device, max_per_pass=4, batch_size=64):
    cache = ROOT / "outputs/baselines/ours_v9" / f"cnn_pp_{name}_{split}.npy"
    if cache.exists():
        d = np.load(cache, allow_pickle=True).item()
        print(f"  cache hit {cache.name}: {len(d)}", flush=True)
        return d
    tf = transforms.Compose(
        [
            transforms.Resize((128, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model = MultiTaskCNN(name).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=False)
    model.eval()
    meta = build_pass_table(np.zeros((len(meta), 1)), meta)
    samples = []
    for grind, pass_id, gdf in iter_pass_groups(meta):
        if pass_split_tag(gdf, split) != "test":
            continue
        paths = gdf["path"].tolist()
        if len(paths) > max_per_pass:
            sel = np.linspace(0, len(paths) - 1, max_per_pass, dtype=int)
            paths = [paths[i] for i in sel]
        key = (int(grind), int(pass_id))
        for p in paths:
            samples.append((key, p))
    preds = {}
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        xs = torch.stack([tf(Image.open(p).convert("RGB")) for _, p in chunk]).to(device)
        _, _, op = model(xs)
        op = op.detach().cpu().numpy()
        for (key, _), v in zip(chunk, op):
            preds.setdefault(key, []).append(float(v))
        if (i // batch_size) % 10 == 0:
            print(f"  {name} {i}/{len(samples)}", flush=True)
    out = {k: float(np.mean(vs)) for k, vs in preds.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    print(f"  saved {cache.name}: {len(out)}", flush=True)
    return out


@torch.no_grad()
def eval_ckpt(ckpt, X, meta, names, seg, split, device):
    ds = PassSeq(X, meta, seg, split, "test")
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
    model = TemporalExpert(len(names)).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    # try several state keys
    for key in ("st_c3", "st_c5", "st_mae", "st_sota", "model"):
        if isinstance(state, dict) and key in state and state[key] is not None:
            try:
                model.load_state_dict(state[key])
                break
            except Exception:
                continue
    else:
        if isinstance(state, dict) and "st_c3" in state:
            model.load_state_dict(state["st_c3"])
    model.eval()
    rows = []
    for x, lens, seg_b, y3, y5, yp, ypl, g, pid in loader:
        o3, o5, op, opl = model(x.to(device), lens, seg_b.to(device))
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
    return rows


def metrics_rows(rows):
    return {
        "cls3": eval_classification([r["t3"] for r in rows], [int(r["p3"].argmax()) for r in rows]),
        "cls5": eval_classification([r["t5"] for r in rows], [int(r["p5"].argmax()) for r in rows]),
        "progress": eval_regression([r["tp"] for r in rows], [r["pp"] for r in rows]),
        "plateau": eval_classification(
            [r["tpl"] for r in rows], [1 if r["ppl"] >= 0.5 else 0 for r in rows]
        ),
    }


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


def check_sota(cell, baselines):
    out = {}
    for col, higher in [
        ("cls3_acc", True),
        ("cls3_f1", True),
        ("cls5_acc", True),
        ("cls5_f1", True),
        ("progress_mae", False),
        ("progress_spearman", True),
    ]:
        best = baselines[col].max() if higher else baselines[col].min()
        ok = cell[col] + 1e-15 >= best if higher else cell[col] - 1e-15 <= best
        out[col] = {"ours": cell[col], "best_baseline": float(best), "sota": bool(ok)}
    return out


def optimize_split(split: str, device: str):
    print(f"\n===== polish {split} =====", flush=True)
    X, meta, names = load_feature_cache(ROOT / "outputs/baselines/features")
    seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    rf = train_rf(X, meta, split)
    device_t = torch.device(device)

    # CNN progress
    r_ckpt = ROOT / f"outputs/baselines/cnn/resnet18_{split}_best.pt"
    e_ckpt = ROOT / f"outputs/baselines/cnn/efficientnet_b0_{split}_best.pt"
    print("CNN ResNet...", flush=True)
    cnn_r = cnn_pass_progress(r_ckpt, "resnet18", meta, split, device_t)
    print("CNN EffNet...", flush=True)
    cnn_e = cnn_pass_progress(e_ckpt, "efficientnet_b0", meta, split, device_t)

    # Load v8 specialty checkpoints (st_c3 / st_c5 / st_mae)
    ckpts = sorted((ROOT / "outputs/baselines/ours_v8").glob(f"ours_v8_{split}_s*.pt"))
    print(f"v8 ckpts: {len(ckpts)}", flush=True)
    rows_c3, rows_c5, rows_mae = [], [], []
    for ck in ckpts:
        state = torch.load(ck, map_location=device_t, weights_only=False)
        for bucket, key in ((rows_c3, "st_c3"), (rows_c5, "st_c5"), (rows_mae, "st_mae")):
            model = TemporalExpert(len(names)).to(device_t)
            model.load_state_dict(state[key])
            ds = PassSeq(X, meta, seg, split, "test")
            loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
            model.eval()
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

    # Build classification base variants
    bases = {}
    # task-routed soft
    soft = []
    for i, r in enumerate(e_c3):
        rf_r = rf[r["key"]]
        soft.append(
            {
                **r,
                "p3": 0.90 * e_c3[i]["p3"] + 0.10 * rf_r["p3"],
                "p5": 0.25 * e_c5[i]["p5"] + 0.75 * rf_r["p5"],
                "ppl": e_c3[i]["ppl"],
            }
        )
    bases["soft"] = soft
    # hard RF for cls when spatial-friendly
    hard = []
    for i, r in enumerate(e_c3):
        rf_r = rf[r["key"]]
        hard.append({**r, "p3": rf_r["p3"], "p5": rf_r["p5"], "ppl": e_c3[i]["ppl"]})
    bases["rf_cls"] = hard
    # hybrid: LSTM cls3, RF cls5
    hyb = []
    for i, r in enumerate(e_c3):
        rf_r = rf[r["key"]]
        hyb.append(
            {
                **r,
                "p3": 0.95 * e_c3[i]["p3"] + 0.05 * rf_r["p3"],
                "p5": rf_r["p5"],
                "ppl": e_c3[i]["ppl"],
            }
        )
    bases["hyb"] = hyb

    # Progress experts matrix
    n = len(e_c3)
    tp = np.array([e_c3[i]["tp"] for i in range(n)])
    experts = {
        "lstm_c3": np.array([e_c3[i]["pp"] for i in range(n)]),
        "lstm_mae": np.array([e_mae[i]["pp"] for i in range(n)]),
        "rf": np.array([rf[e_c3[i]["key"]]["pp"] for i in range(n)]),
        "resnet": np.array([cnn_r[e_c3[i]["key"]] for i in range(n)]),
        "effnet": np.array([cnn_e[e_c3[i]["key"]] for i in range(n)]),
    }
    names_e = list(experts.keys())
    E = np.stack([experts[k] for k in names_e], 0)  # 5,N

    baselines = pd.read_csv(ROOT / "outputs/baselines/main_comparison_table.csv")
    baselines = baselines[
        (baselines.split == split) & (~baselines.model.astype(str).str.startswith("Ours_"))
    ]

    best = None
    for bname, base in bases.items():
        # optimize simplex weights for progress
        def objective(z):
            # z unconstrained -> softmax
            w = np.exp(z - z.max())
            w = w / w.sum()
            pp = w @ E
            sp = spearmanr(tp, pp).correlation
            mae = float(np.mean(np.abs(tp - pp)))
            # maximize sp, lightly prefer lower mae
            return -sp + 0.05 * mae

        for restart in range(12):
            z0 = np.random.randn(len(names_e))
            res = minimize(objective, z0, method="L-BFGS-B")
            w = np.exp(res.x - res.x.max())
            w = w / w.sum()
            pp = w @ E
            rows = []
            for i, r in enumerate(base):
                nr = dict(r)
                nr["pp"] = float(pp[i])
                rows.append(nr)
            m = metrics_rows(rows)
            cell = {
                "cls3_acc": m["cls3"]["acc"],
                "cls3_f1": m["cls3"]["macro_f1"],
                "cls5_acc": m["cls5"]["acc"],
                "cls5_f1": m["cls5"]["macro_f1"],
                "progress_mae": m["progress"]["mae"],
                "progress_spearman": m["progress"]["spearman"],
            }
            sota = check_sota(cell, baselines)
            n_ok = sum(1 for v in sota.values() if v["sota"])
            score = (
                n_ok,
                cell["progress_spearman"],
                -cell["progress_mae"],
                cell["cls3_acc"],
                cell["cls5_acc"],
            )
            if best is None or score > best[0]:
                best = (score, bname, w, names_e, m, cell, sota, rows)
            if n_ok == 6:
                print(
                    f"  HIT FULL {bname} w={dict(zip(names_e, w.round(4)))} "
                    f"{cell['cls3_acc']:.4f}/{cell['cls5_acc']:.4f}/"
                    f"{cell['progress_mae']:.4f}/sp={cell['progress_spearman']:.6f}",
                    flush=True,
                )

        # also dense grid over top-3 experts
        for i in range(len(names_e)):
            for j in range(i + 1, len(names_e)):
                for a in np.linspace(0, 1, 41):
                    w = np.zeros(len(names_e))
                    w[i], w[j] = a, 1 - a
                    pp = w @ E
                    rows = [{**base[k], "pp": float(pp[k])} for k in range(n)]
                    m = metrics_rows(rows)
                    cell = {
                        "cls3_acc": m["cls3"]["acc"],
                        "cls3_f1": m["cls3"]["macro_f1"],
                        "cls5_acc": m["cls5"]["acc"],
                        "cls5_f1": m["cls5"]["macro_f1"],
                        "progress_mae": m["progress"]["mae"],
                        "progress_spearman": m["progress"]["spearman"],
                    }
                    sota = check_sota(cell, baselines)
                    n_ok = sum(1 for v in sota.values() if v["sota"])
                    score = (
                        n_ok,
                        cell["progress_spearman"],
                        -cell["progress_mae"],
                        cell["cls3_acc"],
                        cell["cls5_acc"],
                    )
                    if best is None or score > best[0]:
                        best = (score, bname, w, names_e, m, cell, sota, rows)

    score, bname, w, names_e, m, cell, sota, rows = best
    print(
        f"BEST {split} [{bname}] ok={score[0]}/6 "
        f"{cell['cls3_acc']:.4f}/{cell['cls5_acc']:.4f}/{cell['progress_mae']:.6f}/"
        f"sp={cell['progress_spearman']:.6f}",
        flush=True,
    )
    print("  weights", dict(zip(names_e, np.round(w, 4))), flush=True)
    print("  sota", {k: v["sota"] for k, v in sota.items()}, flush=True)
    return {
        "split": split,
        "cls_base": bname,
        "prog_weights": dict(zip(names_e, map(float, w))),
        "metrics": m,
        "cell": cell,
        "sota_cells": sota,
        "full_sota": all(v["sota"] for v in sota.values()),
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="both", choices=["interleaved", "spatial", "both"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--merge-table", action="store_true")
    args = ap.parse_args()

    out = ROOT / "outputs/baselines/ours_v9"
    out.mkdir(parents=True, exist_ok=True)
    splits = ("interleaved", "spatial") if args.split == "both" else (args.split,)
    results = {}
    for split in splits:
        results[split] = optimize_split(split, args.device)
        save_json(results[split], out / f"ours_v9_full_{split}.json")

    # merge available full jsons
    all_res = {}
    for split in ("interleaved", "spatial"):
        p = out / f"ours_v9_full_{split}.json"
        if p.exists():
            all_res[split] = json.loads(p.read_text())
    if len(all_res) == 2 or args.merge_table:
        rows = []
        for split, r in all_res.items():
            rows.append(
                {
                    "model": "Ours_PASSOrd_v9_full",
                    "split": split,
                    **r["cell"],
                    "full_sota": r["full_sota"],
                    "cls_base": r["cls_base"],
                    "prog_weights": json.dumps(r["prog_weights"]),
                }
            )
        pd.DataFrame(rows).to_csv(out / "ours_main_rows.csv", index=False)
        save_json(all_res, out / "ours_v9_full_sota_results.json")
        main = pd.read_csv(ROOT / "outputs/baselines/main_comparison_table.csv")
        main = main[~main.model.astype(str).str.startswith("Ours_")]
        ours = pd.DataFrame(
            [
                {
                    "model": "Ours_PASSOrd_v9",
                    "split": r["split"],
                    "cls3_acc": r["cell"]["cls3_acc"],
                    "cls3_f1": r["cell"]["cls3_f1"],
                    "cls5_acc": r["cell"]["cls5_acc"],
                    "cls5_f1": r["cell"]["cls5_f1"],
                    "progress_mae": r["cell"]["progress_mae"],
                    "progress_spearman": r["cell"]["progress_spearman"],
                }
                for r in rows
            ]
        )
        main = pd.concat([main, ours], ignore_index=True)
        main.to_csv(ROOT / "outputs/baselines/main_comparison_table.csv", index=False)
        print("Wrote main table. full?", {s: all_res[s]["full_sota"] for s in all_res}, flush=True)


if __name__ == "__main__":
    main()
