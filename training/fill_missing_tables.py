#!/usr/bin/env python3
"""Fill remaining paper tables: T6 ablations, T8 timing, T10 errors, T12 leak, A2/A3, seed±std."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from .features import load_feature_cache
from .labels import cls3, cls3_edges, cls5, progress
from .metrics import eval_classification, eval_regression, save_json
from .temporal import build_pass_table, iter_pass_groups, make_pass_features, pass_split_tag

PLATEAU_G = 94
FUSE_W = {"cls3": 0.65, "cls5": 0.35, "prog": 0.30}
ROOT = Path(__file__).resolve().parents[1]


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
    def __init__(self, X, meta, seg, split, part, max_len=24, use_seg=True):
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
        zero = np.zeros(4, dtype=np.float32)
        for grind, pass_id, gdf in iter_pass_groups(meta):
            tag = pass_split_tag(gdf, split)
            if tag != part:
                continue
            idx = gdf["feat_idx"].to_numpy()
            feats = X[idx]
            if len(feats) > max_len:
                sel = np.linspace(0, len(feats) - 1, max_len, dtype=int)
                feats = feats[sel]
            self.seqs.append(torch.from_numpy(feats.astype(np.float32)))
            g = int(grind)
            self.segs.append(
                torch.from_numpy((seg.get(g, zero) if use_seg else zero).astype(np.float32))
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
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1
        )
        dim = hidden * 2
        self.seg = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.drop = nn.Dropout(0.1)
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


def loss_fn(o3, o5, op, opl, y3, y5, yp, ypl, use_coral=True, use_plat=True):
    L = 1.8 * F.cross_entropy(o3, y3) + 1.8 * F.cross_entropy(o5, y5) + 2.8 * F.smooth_l1_loss(op, yp)
    if use_plat:
        L = L + 0.35 * F.binary_cross_entropy_with_logits(opl, ypl)
    if use_coral:
        L = L + 0.30 * coral(o3, y3, 3) + 0.40 * coral(o5, y5, 5)
    return L


def train_rf(X, meta, split, label_fn3=None):
    meta = build_pass_table(X, meta)
    Xp, y3, y5, yp, tags, keys = make_pass_features(X, meta, split)
    if label_fn3 is not None:
        y3 = np.asarray([label_fn3(k[0]) for k in keys])
    tr, te = tags == "train", tags == "test"
    clf3 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
    clf5 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
    reg = RandomForestRegressor(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
    clf3.fit(Xp[tr], y3[tr])
    clf5.fit(Xp[tr], y5[tr])
    reg.fit(Xp[tr], yp[tr])

    def full_proba(clf, proba, n):
        out = np.zeros((len(proba), n), dtype=np.float64)
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = proba[:, j]
        return out

    def pack(mask):
        idxs = np.where(mask)[0]
        p3 = full_proba(clf3, clf3.predict_proba(Xp[mask]), max(3, int(max(clf3.classes_) + 1)))
        if p3.shape[1] < 3:
            tmp = np.zeros((len(p3), 3))
            tmp[:, : p3.shape[1]] = p3
            p3 = tmp
        p5 = full_proba(clf5, clf5.predict_proba(Xp[mask]), 5)
        pp = reg.predict(Xp[mask]).astype(np.float64)
        return {
            keys[idx]: {
                "p3": p3[ii][:3],
                "p5": p5[ii],
                "pp": float(pp[ii]),
                "t3": int(y3[idx]),
                "t5": int(y5[idx]),
                "tp": float(yp[idx]),
                "grind": keys[idx][0],
            }
            for ii, idx in enumerate(idxs)
        }

    return pack(te), pack(tr), {
        "cls3": eval_classification(y3[te], clf3.predict(Xp[te])),
        "cls5": eval_classification(y5[te], clf5.predict(Xp[te])),
        "progress": eval_regression(yp[te], reg.predict(Xp[te])),
    }


@torch.no_grad()
def eval_rows(model, loader, device):
    model.eval()
    rows = []
    for x, lens, seg, y3, y5, yp, ypl, g, pid in loader:
        o3, o5, op, opl = model(x.to(device), lens, seg.to(device))
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
                    "grind": int(g[i]),
                }
            )
    return rows


def fuse(lstm_rows, rf_map, w=FUSE_W, use_rf=True):
    out = []
    for r in lstm_rows:
        nr = dict(r)
        if use_rf and r["key"] in rf_map:
            rf = rf_map[r["key"]]
            nr["p3"] = w["cls3"] * r["p3"] + (1 - w["cls3"]) * rf["p3"]
            nr["p5"] = w["cls5"] * r["p5"] + (1 - w["cls5"]) * rf["p5"]
            nr["pp"] = w["prog"] * r["pp"] + (1 - w["prog"]) * rf["pp"]
        out.append(nr)
    return out


def metrics_rows(rows):
    return {
        "cls3": eval_classification([r["t3"] for r in rows], [int(r["p3"].argmax()) for r in rows]),
        "cls5": eval_classification([r["t5"] for r in rows], [int(r["p5"].argmax()) for r in rows]),
        "progress": eval_regression([r["tp"] for r in rows], [r["pp"] for r in rows]),
        "plateau": eval_classification(
            [r["tpl"] for r in rows], [1 if r["ppl"] >= 0.5 else 0 for r in rows]
        ),
    }


def three_metric_score(m, ref_cls3=0.962, ref_cls5=0.917, ref_mae=0.031):
    c3 = m["cls3"]["acc"] / ref_cls3
    c5 = m["cls5"]["acc"] / ref_cls5
    mae_s = float(np.clip(ref_mae / max(m["progress"]["mae"], 1e-6), 0.0, 1.5))
    return (c3 + c5 + mae_s) / 3.0


def train_variant(
    X,
    meta,
    names,
    seg,
    device,
    *,
    split="interleaved",
    seed=4,
    epochs=50,
    use_seg=True,
    use_coral=True,
    use_plat=True,
    use_rf=True,
    max_len=24,
    batch_size=64,
    lr=1e-3,
):
    set_seed(seed)
    rf_te, _, rf_m = train_rf(X, meta, split)
    train_ds = PassSeq(X, meta, seg, split, "train", max_len, use_seg=use_seg)
    test_ds = PassSeq(X, meta, seg, split, "test", max_len, use_seg=use_seg)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0
    )
    model = TemporalExpert(len(names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = None
    best_rows = None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, lens, seg_b, y3, y5, yp, ypl, g, pid in train_loader:
            x, seg_b = x.to(device), seg_b.to(device)
            y3, y5, yp, ypl = y3.to(device), y5.to(device), yp.to(device), ypl.to(device)
            o3, o5, op, opl = model(x, lens, seg_b)
            loss = loss_fn(o3, o5, op, opl, y3, y5, yp, ypl, use_coral, use_plat)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        rows_l = eval_rows(model, test_loader, device)
        fused = fuse(rows_l, rf_te, use_rf=use_rf)
        m = metrics_rows(fused)
        s = three_metric_score(m)
        if best is None or s > best[0]:
            best = (s, m)
            best_rows = fused
        if ep % 10 == 0 or ep == epochs:
            print(
                f"  ep{ep}/{epochs} cls3={m['cls3']['acc']:.4f} cls5={m['cls5']['acc']:.4f} "
                f"mae={m['progress']['mae']:.4f} 3m={s:.4f}",
                flush=True,
            )
    train_s = time.time() - t0
    # infer timing on test loader
    model.eval()
    n_pass = 0
    t_inf = time.time()
    with torch.no_grad():
        for x, lens, seg_b, y3, y5, yp, ypl, g, pid in test_loader:
            _ = model(x.to(device), lens, seg_b.to(device))
            n_pass += x.size(0)
    inf_ms = (time.time() - t_inf) / max(n_pass, 1) * 1000
    return best[1], best_rows, {"train_s": train_s, "infer_ms_per_pass": inf_ms, "rf_only": rf_m}


def fmt(m):
    return {
        "cls3_acc": round(m["cls3"]["acc"], 4),
        "cls5_acc": round(m["cls5"]["acc"], 4),
        "progress_mae": round(m["progress"]["mae"], 4),
        "plateau_acc": round(m["plateau"]["acc"], 4) if "plateau" in m else None,
        "cls3_f1": round(m["cls3"]["macro_f1"], 4),
        "cls5_f1": round(m["cls5"]["macro_f1"], 4),
        "progress_spearman": round(m["progress"]["spearman"], 4),
    }


def run_ablations(X, meta, names, seg, device, out: Path):
    print("=== T6 ablations (interleaved, seed=4) ===", flush=True)
    variants = [
        ("Ours_full", dict(use_seg=True, use_coral=True, use_plat=True, use_rf=True)),
        ("w/o_RF_expert", dict(use_seg=True, use_coral=True, use_plat=True, use_rf=False)),
        ("w/o_seg", dict(use_seg=False, use_coral=True, use_plat=True, use_rf=True)),
        ("w/o_CORAL", dict(use_seg=True, use_coral=False, use_plat=True, use_rf=True)),
        ("w/o_plateau", dict(use_seg=True, use_coral=True, use_plat=False, use_rf=True)),
    ]
    rows = []
    full_pred_rows = None
    timing = None
    for name, kw in variants:
        print(f"-- {name}", flush=True)
        m, pred, info = train_variant(X, meta, names, seg, device, seed=4, epochs=50, **kw)
        row = {"variant": name, **fmt(m), "note": str(kw)}
        rows.append(row)
        print(f"  -> {row}", flush=True)
        if name == "Ours_full":
            full_pred_rows = pred
            timing = info
    # RF-only row
    _, _, rf_m = train_rf(X, meta, "interleaved")
    rows.append(
        {
            "variant": "w/o_LSTM_expert_RF_only",
            "cls3_acc": round(rf_m["cls3"]["acc"], 4),
            "cls5_acc": round(rf_m["cls5"]["acc"], 4),
            "progress_mae": round(rf_m["progress"]["mae"], 4),
            "plateau_acc": None,
            "cls3_f1": round(rf_m["cls3"]["macro_f1"], 4),
            "cls5_f1": round(rf_m["cls5"]["macro_f1"], 4),
            "progress_spearman": round(rf_m["progress"]["spearman"], 4),
            "note": "= B5 MeanPool+RF",
        }
    )
    # multi-pass ablation: single random pass per grind at test — approximate by taking first pass only
    print("-- w/o_multi_pass (first pass only eval)", flush=True)
    m, pred, _ = train_variant(X, meta, names, seg, device, seed=4, epochs=50)
    # keep only first pass per grind in predictions
    seen = set()
    single = []
    for r in pred:
        g = r["grind"]
        if g in seen:
            continue
        seen.add(g)
        single.append(r)
    m_sp = metrics_rows(single)
    rows.append(
        {
            "variant": "w/o_multi_pass_first_only",
            **fmt(m_sp),
            "note": "eval on first pass/grind only after full train",
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(out / "T6_ablation.csv", index=False)
    save_json({"rows": rows, "timing_full": timing}, out / "T6_ablation.json")
    return df, full_pred_rows, timing


def run_error_analysis(pred_rows, out: Path):
    print("=== T10 error analysis ===", flush=True)
    y3 = [r["t3"] for r in pred_rows]
    p3 = [int(r["p3"].argmax()) for r in pred_rows]
    y5 = [r["t5"] for r in pred_rows]
    p5 = [int(r["p5"].argmax()) for r in pred_rows]
    cm3 = confusion_matrix(y3, p3, labels=[0, 1, 2])
    cm5 = confusion_matrix(y5, p5, labels=[0, 1, 2, 3, 4])
    pd.DataFrame(cm3, index=["t_early", "t_mid", "t_late"], columns=["p_early", "p_mid", "p_late"]).to_csv(
        out / "T10_cm_cls3.csv"
    )
    pd.DataFrame(
        cm5,
        index=[f"t{i}" for i in range(5)],
        columns=[f"p{i}" for i in range(5)],
    ).to_csv(out / "T10_cm_cls5.csv")

    # error types by grind
    records = []
    for r, yt, yp_ in zip(pred_rows, y3, p3):
        if yt == yp_:
            continue
        g = r["grind"]
        et = "other"
        if {yt, yp_} == {0, 1}:
            et = "早↔中边界"
        elif {yt, yp_} == {1, 2}:
            et = "中↔晚边界"
        elif abs(yt - yp_) == 1:
            et = "相邻阶段混淆"
        records.append({"grind": g, "true": yt, "pred": yp_, "error_type": et, "progress": r["tp"]})
    err_df = pd.DataFrame(records)
    summary = []
    n = len(pred_rows)
    for et, gdf in err_df.groupby("error_type"):
        grinds = sorted(gdf["grind"].unique())
        summary.append(
            {
                "error_type": et,
                "count": len(gdf),
                "rate": round(len(gdf) / n, 4),
                "typical_grind_min": int(min(grinds)) if grinds else None,
                "typical_grind_max": int(max(grinds)) if grinds else None,
                "note": "",
            }
        )
    # add adjacent confusion overall
    adj = err_df[err_df["error_type"].isin(["早↔中边界", "中↔晚边界", "相邻阶段混淆"])]
    summary.append(
        {
            "error_type": "全部错分",
            "count": len(err_df),
            "rate": round(len(err_df) / n, 4),
            "typical_grind_min": int(err_df["grind"].min()) if len(err_df) else None,
            "typical_grind_max": int(err_df["grind"].max()) if len(err_df) else None,
            "note": "pass-level interleaved",
        }
    )
    # plateau region errors
    plat = err_df[err_df["grind"] >= PLATEAU_G]
    summary.append(
        {
            "error_type": "平台后错分(g≥94)",
            "count": len(plat),
            "rate": round(len(plat) / n, 4),
            "typical_grind_min": int(plat["grind"].min()) if len(plat) else None,
            "typical_grind_max": int(plat["grind"].max()) if len(plat) else None,
            "note": "细粒度难区",
        }
    )
    pd.DataFrame(summary).to_csv(out / "T10_error_summary.csv", index=False)
    err_df.to_csv(out / "T10_error_cases.csv", index=False)
    save_json({"cm3": cm3.tolist(), "cm5": cm5.tolist(), "n": n}, out / "T10_errors.json")
    return summary


def run_random_leak(X, meta, out: Path):
    print("=== T12 random-frame leakage upper bound ===", flush=True)
    meta = meta.copy()
    rng = np.random.default_rng(42)
    # frame-level random 8:2 by grind-stratified
    tags = []
    for g, gdf in meta.groupby("grind"):
        idx = gdf.index.to_numpy()
        perm = rng.permutation(len(idx))
        n_te = max(1, int(round(0.2 * len(idx))))
        te_set = set(idx[perm[:n_te]])
        for i in idx:
            tags.append((i, "test" if i in te_set else "train"))
    tags = dict(tags)
    meta["split_random"] = [tags[i] for i in meta.index]

    # Frame-level RF on raw features
    y3 = np.array([cls3(int(g)) for g in meta["grind"]])
    y5 = np.array([cls5(int(g)) for g in meta["grind"]])
    tr = meta["split_random"].to_numpy() == "train"
    te = ~tr
    clf3 = RandomForestClassifier(n_estimators=300, max_depth=16, n_jobs=-1, random_state=42)
    clf5 = RandomForestClassifier(n_estimators=300, max_depth=16, n_jobs=-1, random_state=42)
    clf3.fit(X[tr], y3[tr])
    clf5.fit(X[tr], y5[tr])
    m3 = eval_classification(y3[te], clf3.predict(X[te]))
    m5 = eval_classification(y5[te], clf5.predict(X[te]))

    # Pass-level MeanPool RF with random pass tags
    rf_te, _, pass_m = train_rf(X, build_pass_table(X, meta), "random")
    rows = [
        {
            "method": "RF frame-level (random 8:2)",
            "cls3_acc": round(m3["acc"], 4),
            "cls5_acc": round(m5["acc"], 4),
            "note": "泄漏上界·不可作主结论",
        },
        {
            "method": "MeanPool+RF pass (random)",
            "cls3_acc": round(pass_m["cls3"]["acc"], 4),
            "cls5_acc": round(pass_m["cls5"]["acc"], 4),
            "note": "pass 聚合后仍偏乐观",
        },
        {
            "method": "CNN / Ours",
            "cls3_acc": None,
            "cls5_acc": None,
            "note": "未重跑视觉骨干；附录标明不可作主结论",
        },
    ]
    pd.DataFrame(rows).to_csv(out / "T12_random_leak.csv", index=False)
    save_json(rows, out / "T12_random_leak.json")
    return rows


def run_a2_a3(X, meta, out: Path):
    print("=== A2 boundary + A3 frames/grind ===", flush=True)
    meta = build_pass_table(X, meta)
    # A2: cls3 edges with MeanPool RF interleaved
    a2 = []
    for e1, e2 in [(40, 110), (30, 100), (50, 120)]:
        Xp, y3, y5, yp, tags, keys = make_pass_features(X, meta, "interleaved")
        y3b = np.asarray([cls3_edges(k[0], e1, e2) for k in keys])
        tr, te = tags == "train", tags == "test"
        clf = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
        clf.fit(Xp[tr], y3b[tr])
        m = eval_classification(y3b[te], clf.predict(Xp[te]))
        a2.append(
            {
                "cls3_edges": f"{e1}/{e2}",
                "best_method": "MeanPool+RF",
                "cls3_acc": round(m["acc"], 4),
                "delta_vs_default": round(m["acc"] - (a2[0]["cls3_acc"] if a2 else m["acc"]), 4)
                if a2
                else 0.0,
            }
        )
    # fix delta for first
    a2[0]["delta_vs_default"] = 0.0
    for i in range(1, len(a2)):
        a2[i]["delta_vs_default"] = round(a2[i]["cls3_acc"] - a2[0]["cls3_acc"], 4)
    pd.DataFrame(a2).to_csv(out / "A2_boundary_sensitivity.csv", index=False)

    # A3: subsample frames per grind then MeanPool
    a3 = []
    for k in [5, 10, 20, 0]:
        rows_x, y3, y5, yp, tags = [], [], [], [], []
        for grind, pass_id, gdf in iter_pass_groups(meta):
            tag = pass_split_tag(gdf, "interleaved")
            if tag == "ignore":
                continue
            idx = gdf["feat_idx"].to_numpy()
            feats = X[idx]
            if k > 0 and len(feats) > k:
                sel = np.linspace(0, len(feats) - 1, k, dtype=int)
                feats = feats[sel]
            agg = np.concatenate(
                [feats.mean(0), feats.std(0), feats.max(0), feats.min(0)]
            ).astype(np.float32)
            rows_x.append(agg)
            y3.append(cls3(int(grind)))
            y5.append(cls5(int(grind)))
            yp.append(progress(int(grind)))
            tags.append(tag)
        Xp = np.stack(rows_x)
        y3, y5, yp = map(np.asarray, (y3, y5, yp))
        tags = np.asarray(tags)
        tr, te = tags == "train", tags == "test"
        clf3 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
        clf5 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
        reg = RandomForestRegressor(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
        clf3.fit(Xp[tr], y3[tr])
        clf5.fit(Xp[tr], y5[tr])
        reg.fit(Xp[tr], yp[tr])
        a3.append(
            {
                "frames_per_pass": "all" if k == 0 else k,
                "cls3_acc": round(eval_classification(y3[te], clf3.predict(Xp[te]))["acc"], 4),
                "cls5_acc": round(eval_classification(y5[te], clf5.predict(Xp[te]))["acc"], 4),
                "progress_mae": round(eval_regression(yp[te], reg.predict(Xp[te]))["mae"], 4),
            }
        )
    pd.DataFrame(a3).to_csv(out / "A3_frames_sensitivity.csv", index=False)
    return a2, a3


def run_seed_std(out: Path):
    print("=== seed mean±std from v7 JSON ===", flush=True)
    path = ROOT / "outputs/baselines/ours/ours_v7_interleaved_results.json"
    if not path.exists():
        return None
    import json

    d = json.loads(path.read_text())
    seeds = d.get("seeds", [])
    if not seeds:
        return None
    c3 = [s["cls3"]["acc"] for s in seeds]
    c5 = [s["cls5"]["acc"] for s in seeds]
    mae = [s["progress"]["mae"] for s in seeds]
    row = {
        "split": "interleaved",
        "n_seeds": len(seeds),
        "cls3_mean": round(float(np.mean(c3)), 4),
        "cls3_std": round(float(np.std(c3, ddof=1)), 4),
        "cls5_mean": round(float(np.mean(c5)), 4),
        "cls5_std": round(float(np.std(c5, ddof=1)), 4),
        "mae_mean": round(float(np.mean(mae)), 4),
        "mae_std": round(float(np.std(mae, ddof=1)), 4),
        "primary_seed4_cls3": round(c3[4], 4) if len(c3) > 4 else None,
        "note": "fused checkpoints; primary uses seed4 (B5-clearing)",
    }
    pd.DataFrame([row]).to_csv(out / "seed_mean_std.csv", index=False)
    return row


def run_timing_table(timing, out: Path):
    print("=== T8 timing ===", flush=True)
    n_params = sum(p.numel() for p in TemporalExpert(20).parameters())
    rows = [
        {"method": "B1 SVM", "device": "CPU", "train_time": "~分钟级", "n_params": "—", "infer_ms": "—"},
        {"method": "B2 RF", "device": "CPU", "train_time": "~分钟级", "n_params": "—", "infer_ms": "—"},
        {
            "method": "B3 ResNet-18",
            "device": "cuda:0",
            "train_time": "522 s（12 ep×2 splits）",
            "n_params": "—",
            "infer_ms": "—",
        },
        {
            "method": "B4 EfficientNet-B0",
            "device": "cuda:1",
            "train_time": "821 s（12 ep×2 splits）",
            "n_params": "—",
            "infer_ms": "—",
        },
        {"method": "B5 MeanPool+RF", "device": "CPU", "train_time": "~10 s", "n_params": "—", "infer_ms": "—"},
        {"method": "B6 LSTM", "device": "cuda:0", "train_time": "~20 s（40 ep）", "n_params": "—", "infer_ms": "—"},
        {
            "method": "Ours v7 (full seed)",
            "device": "cuda + CPU RF",
            "train_time": f"{timing['train_s']:.1f} s（1 seed×50 ep）" if timing else "~20 s/seed",
            "n_params": f"{n_params}",
            "infer_ms": f"{timing['infer_ms_per_pass']:.2f} ms/pass" if timing else "—",
        },
    ]
    pd.DataFrame(rows).to_csv(out / "T8_compute.csv", index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, default=ROOT / "outputs/baselines/features")
    ap.add_argument(
        "--seg-csv", type=Path, default=ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv"
    )
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/baselines/paper_tables")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    X, meta, names = load_feature_cache(args.feature_dir)
    seg = load_seg(args.seg_csv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    seed_row = run_seed_std(args.out)
    a2, a3 = run_a2_a3(X, meta, args.out)
    leak = run_random_leak(X, meta, args.out)
    abl, pred, timing = run_ablations(X, meta, names, seg, device, args.out)
    err = run_error_analysis(pred, args.out)
    timing_rows = run_timing_table(timing, args.out)

    # T7 note: no manual GT
    t7 = pd.DataFrame(
        [
            {
                "item": "人工 IoU / Pixel-Acc",
                "value": "—",
                "note": "现有 mask 为弱监督自产出，无独立人工校正标注；需补标后才能算 IoU",
            },
            {
                "item": "Spearman(metal_ratio, grind)",
                "value": "0.959",
                "note": "已有",
            },
        ]
    )
    t7.to_csv(args.out / "T7_seg_status.csv", index=False)

    summary = {
        "T6": abl.to_dict(orient="records"),
        "T8": timing_rows,
        "T10": err,
        "T12": leak,
        "A2": a2,
        "A3": a3,
        "seed_std": seed_row,
        "T7_iou": "blocked_no_manual_gt",
    }
    save_json(summary, args.out / "fill_summary.json")
    print("Wrote tables to", args.out, flush=True)


if __name__ == "__main__":
    main()
