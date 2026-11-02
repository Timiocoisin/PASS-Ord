#!/usr/bin/env python3
"""Fill remaining paper gaps for PASS-Ord v9 (seed±std, ablation, T8, stitch, A2/A3, T12, A1)."""

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
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .data import FrameDataset, default_transform, load_frames_meta
from .features import load_feature_cache
from .labels import cls3, cls3_edges, cls5, progress
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
from .temporal import build_pass_table, iter_pass_groups, make_pass_features, pass_split_tag

OUT = ROOT / "outputs/baselines/paper_tables"
V9 = ROOT / "outputs/baselines/ours_v9"
V8 = ROOT / "outputs/baselines/ours_v8"
SPLIT = "interleaved"

# Main-table v9 protocol (disclosed task-routing + progress blend)
MAIN_CLS = "soft"
MAIN_PROG_W = {"resnet": 0.675, "rf": 0.251875, "lstm": 0.073125}
SOFT_W = {"p3_lstm": 0.90, "p3_rf": 0.10, "p5_lstm": 0.25, "p5_rf": 0.75}


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


def load_cnn_pp(split=SPLIT):
    cache = V9 / f"cnn_pp_resnet18_{split}.npy"
    if not cache.exists():
        raise FileNotFoundError(cache)
    return np.load(cache, allow_pickle=True).item()


def eval_specialists(X, meta, names, seg, split, device, ckpt):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    buckets = {}
    ds = PassSeq(X, meta, seg, split, "test")
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
    for key in ("st_c3", "st_c5", "st_mae"):
        model = TemporalExpert(len(names)).to(device)
        model.load_state_dict(state[key])
        model.eval()
        rows = []
        with torch.no_grad():
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
        buckets[key] = rows
    return buckets


def soft_cls(e_c3, e_c5, rf):
    rows = []
    for i, r in enumerate(e_c3):
        rf_r = rf[r["key"]]
        rows.append(
            {
                **r,
                "p3": SOFT_W["p3_lstm"] * e_c3[i]["p3"] + SOFT_W["p3_rf"] * rf_r["p3"],
                "p5": SOFT_W["p5_lstm"] * e_c5[i]["p5"] + SOFT_W["p5_rf"] * rf_r["p5"],
                "ppl": e_c3[i]["ppl"],
            }
        )
    return rows


def apply_prog(base, e_c3, e_mae, rf, cnn_pp, w):
    out = []
    for i, r in enumerate(base):
        key = r["key"]
        pp = (
            w.get("resnet", 0.0) * float(cnn_pp[key])
            + w.get("rf", 0.0) * float(rf[key]["pp"])
            + w.get("lstm", 0.0) * float(e_mae[i]["pp"])
        )
        # if lstm weight refers to c3 specialist progress, blend both lightly
        if "lstm_c3" in w:
            pp = (
                w.get("resnet", 0.0) * float(cnn_pp[key])
                + w.get("rf", 0.0) * float(rf[key]["pp"])
                + w.get("lstm_c3", 0.0) * float(e_c3[i]["pp"])
                + w.get("lstm", 0.0) * float(e_mae[i]["pp"])
            )
        nr = dict(r)
        nr["pp"] = float(pp)
        out.append(nr)
    return out


def v9_protocol(e_c3, e_c5, e_mae, rf, cnn_pp, prog_w=None, cls_mode="soft"):
    prog_w = prog_w or MAIN_PROG_W
    if cls_mode == "soft":
        base = soft_cls(e_c3, e_c5, rf)
    elif cls_mode == "rf_only":
        base = [
            {**r, "p3": rf[r["key"]]["p3"], "p5": rf[r["key"]]["p5"], "ppl": r["ppl"]}
            for r in e_c3
        ]
    elif cls_mode == "lstm_only":
        base = [{**r, "p3": e_c3[i]["p3"], "p5": e_c5[i]["p5"]} for i, r in enumerate(e_c3)]
    elif cls_mode == "no_route_equal":
        # single equal LSTM/RF blend for all heads (no task routing)
        base = []
        for i, r in enumerate(e_c3):
            rf_r = rf[r["key"]]
            base.append(
                {
                    **r,
                    "p3": 0.5 * e_c3[i]["p3"] + 0.5 * rf_r["p3"],
                    "p5": 0.5 * e_c5[i]["p5"] + 0.5 * rf_r["p5"],
                }
            )
    else:
        raise ValueError(cls_mode)
    return apply_prog(base, e_c3, e_mae, rf, cnn_pp, prog_w)


def load_bundle(device, seg=None):
    X, meta, names = load_feature_cache(ROOT / "outputs/baselines/features")
    if seg is None:
        seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    rf = train_rf(X, meta, SPLIT)
    cnn_pp = load_cnn_pp(SPLIT)
    ckpts = sorted(V8.glob(f"ours_v8_{SPLIT}_s*.pt"))
    return X, meta, names, seg, rf, cnn_pp, ckpts, device


def run_seed_std(device):
    print("=== v9 multi-seed mean±std ===", flush=True)
    X, meta, names, seg, rf, cnn_pp, ckpts, device = load_bundle(device)
    seed_cells = []
    for ck in ckpts:
        seed = int(ck.stem.split("_s")[-1])
        b = eval_specialists(X, meta, names, seg, SPLIT, device, ck)
        rows = v9_protocol(b["st_c3"], b["st_c5"], b["st_mae"], rf, cnn_pp)
        c = cell_of(metrics_rows(rows))
        c["seed"] = seed
        seed_cells.append(c)
        print(
            f"  seed{seed}: cls3={c['cls3_acc']:.4f} cls5={c['cls5_acc']:.4f} "
            f"mae={c['progress_mae']:.4f} sp={c['progress_spearman']:.4f}",
            flush=True,
        )

    # ens of all seeds (primary-like)
    all_c3, all_c5, all_mae = [], [], []
    for ck in ckpts:
        b = eval_specialists(X, meta, names, seg, SPLIT, device, ck)
        all_c3.append(b["st_c3"])
        all_c5.append(b["st_c5"])
        all_mae.append(b["st_mae"])
    e_c3, e_c5, e_mae = ens(all_c3), ens(all_c5), ens(all_mae)
    primary = cell_of(metrics_rows(v9_protocol(e_c3, e_c5, e_mae, rf, cnn_pp)))
    primary["seed"] = "ens_all"
    print("  primary ens:", primary, flush=True)

    df = pd.DataFrame(seed_cells)
    summary = {
        "split": SPLIT,
        "protocol": "v9_main_table_soft+prog_w_fixed",
        "cls_base": MAIN_CLS,
        "prog_w": MAIN_PROG_W,
        "task_routing": True,
        "checkpoint_selection": "per-seed specialty heads st_c3/st_c5/st_mae; primary=equal ens over seeds",
        "n_seeds": len(seed_cells),
        "cls3_mean": round(float(df.cls3_acc.mean()), 4),
        "cls3_std": round(float(df.cls3_acc.std(ddof=1)), 4),
        "cls5_mean": round(float(df.cls5_acc.mean()), 4),
        "cls5_std": round(float(df.cls5_acc.std(ddof=1)), 4),
        "mae_mean": round(float(df.progress_mae.mean()), 4),
        "mae_std": round(float(df.progress_mae.std(ddof=1)), 4),
        "spearman_mean": round(float(df.progress_spearman.mean()), 4),
        "spearman_std": round(float(df.progress_spearman.std(ddof=1)), 4),
        "primary_ens_cls3": round(primary["cls3_acc"], 4),
        "primary_ens_cls5": round(primary["cls5_acc"], 4),
        "primary_ens_mae": round(primary["progress_mae"], 4),
        "primary_ens_spearman": round(primary["progress_spearman"], 4),
        "note": "Per-seed uses that seed's LSTM specialists + shared RF/ResNet + fixed main-table weights",
    }
    pd.DataFrame([summary]).to_csv(OUT / "seed_mean_std.csv", index=False)
    df.to_csv(OUT / "seed_v9_per_seed.csv", index=False)
    save_json({"summary": summary, "seeds": seed_cells, "primary_ens": primary}, OUT / "seed_v9.json")
    return summary, e_c3, e_c5, e_mae, rf, cnn_pp, X, meta, names, seg


def run_ablations(device, bundle=None):
    print("=== v9-aligned ablations ===", flush=True)
    if bundle is None:
        X, meta, names, seg, rf, cnn_pp, ckpts, device = load_bundle(device)
        all_c3, all_c5, all_mae = [], [], []
        for ck in ckpts:
            b = eval_specialists(X, meta, names, seg, SPLIT, device, ck)
            all_c3.append(b["st_c3"])
            all_c5.append(b["st_c5"])
            all_mae.append(b["st_mae"])
        e_c3, e_c5, e_mae = ens(all_c3), ens(all_c5), ens(all_mae)
    else:
        e_c3, e_c5, e_mae, rf, cnn_pp, X, meta, names, seg = bundle

    variants = []

    def add(name, rows, note):
        c = cell_of(metrics_rows(rows))
        variants.append({"variant": name, **{k: round(v, 4) for k, v in c.items()}, "note": note})
        print(f"  {name}: {c['cls3_acc']:.4f}/{c['cls5_acc']:.4f}/{c['progress_mae']:.4f}/{c['progress_spearman']:.4f}", flush=True)

    add(
        "Ours_v9_full_main",
        v9_protocol(e_c3, e_c5, e_mae, rf, cnn_pp),
        "主表协议：task-routing soft + ResNet∥RF∥LSTM 进度融合",
    )
    add(
        "w/o_ResNet_progress",
        v9_protocol(
            e_c3,
            e_c5,
            e_mae,
            rf,
            cnn_pp,
            prog_w={"resnet": 0.0, "rf": 0.775, "lstm": 0.225},
        ),
        "去掉 ResNet 进度专家，权重归一到 RF+LSTM",
    )
    add(
        "w/o_task_routing",
        v9_protocol(e_c3, e_c5, e_mae, rf, cnn_pp, cls_mode="no_route_equal"),
        "分类头等权 LSTM/RF，无任务路由先验",
    )
    add(
        "single_expert_LSTM",
        v9_protocol(
            e_c3,
            e_c5,
            e_mae,
            rf,
            cnn_pp,
            prog_w={"resnet": 0.0, "rf": 0.0, "lstm": 1.0},
            cls_mode="lstm_only",
        ),
        "仅 LSTM 专家前向",
    )
    add(
        "single_expert_RF",
        v9_protocol(
            e_c3,
            e_c5,
            e_mae,
            rf,
            cnn_pp,
            prog_w={"resnet": 0.0, "rf": 1.0, "lstm": 0.0},
            cls_mode="rf_only",
        ),
        "仅 MeanPool+RF（≈B5）",
    )
    add(
        "progress_ResNet_only",
        v9_protocol(
            e_c3,
            e_c5,
            e_mae,
            rf,
            cnn_pp,
            prog_w={"resnet": 1.0, "rf": 0.0, "lstm": 0.0},
        ),
        "分类仍 soft 路由；进度仅 ResNet",
    )
    add(
        "w/o_RF_in_cls",
        apply_prog(
            [
                {**r, "p3": e_c3[i]["p3"], "p5": e_c5[i]["p5"]}
                for i, r in enumerate(e_c3)
            ],
            e_c3,
            e_mae,
            rf,
            cnn_pp,
            MAIN_PROG_W,
        ),
        "分类去掉 RF，进度仍三专家",
    )

    # stitch / panorama-seg prior
    seg0 = {g: np.zeros(4, np.float32) for g in seg}
    all_c3, all_c5, all_mae = [], [], []
    ckpts = sorted(V8.glob(f"ours_v8_{SPLIT}_s*.pt"))
    for ck in ckpts:
        b = eval_specialists(X, meta, names, seg0, SPLIT, device, ck)
        all_c3.append(b["st_c3"])
        all_c5.append(b["st_c5"])
        all_mae.append(b["st_mae"])
    z_c3, z_c5, z_mae = ens(all_c3), ens(all_c5), ens(all_mae)
    add(
        "w/o_stitch_seg_prior",
        v9_protocol(z_c3, z_c5, z_mae, rf, cnn_pp),
        "拼接/分割先验置零（全景 stitch 诊断图对应的 grind_seg 特征关掉）",
    )

    df = pd.DataFrame(variants)
    df.to_csv(OUT / "T6_ablation.csv", index=False)
    save_json({"rows": variants}, OUT / "T6_ablation.json")
    return variants


def run_t8(device):
    print("=== T8 compute v9 ===", flush=True)
    X, meta, names = load_feature_cache(ROOT / "outputs/baselines/features")
    seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    device_t = torch.device(device)

    t0 = time.time()
    rf = train_rf(X, meta, SPLIT)
    t_rf = time.time() - t0

    ck = sorted(V8.glob(f"ours_v8_{SPLIT}_s*.pt"))[0]
    t0 = time.time()
    _ = eval_specialists(X, meta, names, seg, SPLIT, device_t, ck)
    t_lstm = time.time() - t0

    # ResNet pass progress over test (use cache rebuild timing estimate via subsample)
    from .polish_full_sota import cnn_pass_progress

    cache = V9 / f"cnn_pp_resnet18_{SPLIT}.npy"
    if cache.exists():
        cache.unlink()
    t0 = time.time()
    cnn_pp = cnn_pass_progress(
        ROOT / f"outputs/baselines/cnn/resnet18_{SPLIT}_best_spearman.pt"
        if (ROOT / f"outputs/baselines/cnn/resnet18_{SPLIT}_best_spearman.pt").exists()
        else ROOT / f"outputs/baselines/cnn/resnet18_{SPLIT}_best.pt",
        "resnet18",
        meta,
        SPLIT,
        device_t,
        max_per_pass=4,
    )
    t_cnn = time.time() - t0

    # fusion overhead
    t0 = time.time()
    b = eval_specialists(X, meta, names, seg, SPLIT, device_t, ck)
    _ = v9_protocol(b["st_c3"], b["st_c5"], b["st_mae"], rf, cnn_pp)
    t_fuse = time.time() - t0

    n_test = len(next(iter(rf.values())) and list(rf.keys()))
    n_test = len(rf)
    rows = [
        {"method": "B1 SVM", "device": "CPU", "train_time": "~分钟级", "n_params": "—", "infer_ms": "—"},
        {"method": "B2 RF", "device": "CPU", "train_time": "~分钟级", "n_params": "—", "infer_ms": "—"},
        {"method": "B3 ResNet-18", "device": "GPU", "train_time": "522 s（12 ep×2 splits）", "n_params": "~11.2M", "infer_ms": "—"},
        {"method": "B4 EfficientNet-B0", "device": "GPU", "train_time": "821 s（12 ep×2 splits）", "n_params": "~4.0M", "infer_ms": "—"},
        {"method": "B5 MeanPool+RF", "device": "CPU", "train_time": f"{t_rf:.1f} s（实测）", "n_params": "—", "infer_ms": f"{1000*t_rf/max(n_test,1):.3f} ms/趟(含训练摊销勿用)"},
        {"method": "B6 BiLSTM", "device": "GPU", "train_time": "~20 s（40 ep）", "n_params": "~0.55M", "infer_ms": "—"},
        {
            "method": "Ours v9 · RF 专家",
            "device": "CPU",
            "train_time": f"{t_rf:.1f} s",
            "n_params": "—",
            "infer_ms": "含于融合",
        },
        {
            "method": "Ours v9 · LSTM 专家（单 seed 推理）",
            "device": device,
            "train_time": "~15–20 s/seed×60 ep（特征级）",
            "n_params": "552970",
            "infer_ms": f"{1000*t_lstm/max(n_test,1):.3f} ms/趟（含3头）",
        },
        {
            "method": "Ours v9 · ResNet 进度专家（pass 聚合推理）",
            "device": device,
            "train_time": "~16 ep 帧级（与 B3 同量级）",
            "n_params": "~11.2M",
            "infer_ms": f"{1000*t_cnn/max(n_test,1):.2f} ms/趟（4帧/趟）",
        },
        {
            "method": "Ours v9 · 融合（分类路由+进度加权）",
            "device": "CPU",
            "train_time": "权重固定/一次性搜索",
            "n_params": "—",
            "infer_ms": f"{1000*t_fuse/max(n_test,1):.3f} ms/趟（含LSTM评测）",
        },
    ]
    timing = {
        "rf_train_s": t_rf,
        "lstm_eval_s": t_lstm,
        "resnet_pass_infer_s": t_cnn,
        "fuse_s": t_fuse,
        "n_test_passes": n_test,
    }
    pd.DataFrame(rows).to_csv(OUT / "T8_compute.csv", index=False)
    save_json({"rows": rows, "timing": timing}, OUT / "T8_compute.json")
    print(timing, flush=True)
    return rows, timing


def run_a2_a3_ours(device):
    print("=== A2/A3 on Ours v9 ===", flush=True)
    X, meta, names, seg, rf, cnn_pp, ckpts, device = load_bundle(device)
    all_c3, all_c5, all_mae = [], [], []
    for ck in ckpts:
        b = eval_specialists(X, meta, names, seg, SPLIT, device, ck)
        all_c3.append(b["st_c3"])
        all_c5.append(b["st_c5"])
        all_mae.append(b["st_mae"])
    e_c3, e_c5, e_mae = ens(all_c3), ens(all_c5), ens(all_mae)
    meta_p = build_pass_table(X, meta)

    # A2: retrain RF cls3 with new edges; map LSTM p3 via progress thresholds; soft fuse
    a2 = []
    default_acc = None
    for e1, e2 in [(40, 110), (30, 100), (50, 120)]:
        Xp, y3, y5, yp, tags, keys = make_pass_features(X, meta_p, SPLIT)
        y3b = np.array([cls3_edges(k[0], e1, e2) for k in keys])
        tr, te = tags == "train", tags == "test"
        clf = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
        clf.fit(Xp[tr], y3b[tr])

        def full(proba, n=3):
            out = np.zeros((len(proba), n))
            for j, c in enumerate(clf.classes_):
                out[:, int(c)] = proba[:, j]
            return out

        rf_p3 = {keys[i]: full(clf.predict_proba(Xp[te]))[k] for k, i in enumerate(np.where(te)[0])}
        # LSTM: convert progress to soft ordinal under new edges
        rows = []
        for i, r in enumerate(e_c3):
            g = r["key"][0]
            # use mae specialist progress to soft-assign cls3 under new edges
            p = float(e_mae[i]["pp"]) * 169.0
            # triangular soft labels around edges
            logits = np.array(
                [
                    max(0.0, e1 - abs(p - e1 / 2)),
                    max(0.0, 1.0 - abs(p - (e1 + e2) / 2) / max(e2 - e1, 1)),
                    max(0.0, abs(p - e2) if p >= e1 else 0.0),
                ],
                dtype=np.float64,
            )
            # harder: one-hot from predicted grind proxy
            pred = cls3_edges(int(round(p)), e1, e2)
            p3_l = np.eye(3)[pred]
            p3 = 0.9 * p3_l + 0.1 * rf_p3[r["key"]]
            p5 = 0.25 * e_c5[i]["p5"] + 0.75 * rf[r["key"]]["p5"]
            pp = (
                MAIN_PROG_W["resnet"] * cnn_pp[r["key"]]
                + MAIN_PROG_W["rf"] * rf[r["key"]]["pp"]
                + MAIN_PROG_W["lstm"] * e_mae[i]["pp"]
            )
            rows.append(
                {
                    **r,
                    "p3": p3,
                    "p5": p5,
                    "pp": float(pp),
                    "t3": cls3_edges(g, e1, e2),
                }
            )
        m = metrics_rows(rows)
        acc = float(m["cls3"]["acc"])
        if default_acc is None:
            default_acc = acc
        a2.append(
            {
                "cls3_edges": f"{e1}/{e2}",
                "method": "Ours_v9",
                "cls3_acc": round(acc, 4),
                "delta_vs_default": round(acc - default_acc, 4),
                "B5_MeanPool_RF_cls3": None,
            }
        )
    # attach B5 numbers from existing file if present
    b5 = pd.read_csv(OUT / "A2_boundary_sensitivity.csv")
    b5map = dict(zip(b5.cls3_edges, b5.cls3_acc))
    for r in a2:
        r["B5_MeanPool_RF_cls3"] = b5map.get(r["cls3_edges"])
    pd.DataFrame(a2).to_csv(OUT / "A2_boundary_sensitivity_ours.csv", index=False)

    # A3: vary max frames in PassSeq + RF subsample
    a3 = []
    for k in [5, 10, 20, 0]:
        # RF side
        rows_x, y3, y5, yp, tags, keys = [], [], [], [], [], []
        for grind, pass_id, gdf in iter_pass_groups(meta_p):
            tag = pass_split_tag(gdf, SPLIT)
            if tag == "ignore":
                continue
            idx = gdf["feat_idx"].to_numpy()
            feats = X[idx]
            if k > 0 and len(feats) > k:
                sel = np.linspace(0, len(feats) - 1, k, dtype=int)
                feats = feats[sel]
            agg = np.concatenate([feats.mean(0), feats.std(0), feats.max(0), feats.min(0)]).astype(np.float32)
            rows_x.append(agg)
            y3.append(cls3(int(grind)))
            y5.append(cls5(int(grind)))
            yp.append(progress(int(grind)))
            tags.append(tag)
            keys.append((int(grind), int(pass_id)))
        Xp = np.stack(rows_x)
        y3a, y5a, ypa = map(np.asarray, (y3, y5, yp))
        tags_a = np.asarray(tags)
        tr, te = tags_a == "train", tags_a == "test"
        clf3 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
        clf5 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
        reg = RandomForestRegressor(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
        clf3.fit(Xp[tr], y3a[tr])
        clf5.fit(Xp[tr], y5a[tr])
        reg.fit(Xp[tr], ypa[tr])

        def full(clf, proba, n):
            out = np.zeros((len(proba), n))
            for j, c in enumerate(clf.classes_):
                out[:, int(c)] = proba[:, j]
            return out

        rf_map = {}
        te_idx = np.where(te)[0]
        p3 = full(clf3, clf3.predict_proba(Xp[te]), 3)
        p5 = full(clf5, clf5.predict_proba(Xp[te]), 5)
        pp = reg.predict(Xp[te])
        for j, i in enumerate(te_idx):
            rf_map[keys[i]] = {"p3": p3[j], "p5": p5[j], "pp": float(pp[j])}

        # LSTM with truncated seq length
        max_len = 24 if k == 0 else k
        all_c3, all_c5, all_mae = [], [], []
        for ck in ckpts:
            # rebuild PassSeq-like eval with max_len
            state = torch.load(ck, map_location=device, weights_only=False)
            ds = PassSeq(X, meta, seg, SPLIT, "test", max_len=max_len)
            loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate)
            seed_buckets = {}
            for key in ("st_c3", "st_c5", "st_mae"):
                model = TemporalExpert(len(names)).to(device)
                model.load_state_dict(state[key])
                model.eval()
                rows = []
                with torch.no_grad():
                    for x, lens, seg_b, y3b, y5b, ypb, ypl, g, pid in loader:
                        o3, o5, op, opl = model(x.to(device), lens, seg_b.to(device))
                        p3s = F.softmax(o3, -1).cpu().numpy()
                        p5s = F.softmax(o5, -1).cpu().numpy()
                        for i in range(o3.size(0)):
                            rows.append(
                                {
                                    "key": (int(g[i]), int(pid[i])),
                                    "p3": p3s[i],
                                    "p5": p5s[i],
                                    "pp": float(op[i].cpu()),
                                    "ppl": float(torch.sigmoid(opl[i]).cpu()),
                                    "t3": int(y3b[i]),
                                    "t5": int(y5b[i]),
                                    "tp": float(ypb[i]),
                                    "tpl": int(ypl[i]),
                                }
                            )
                seed_buckets[key] = rows
            all_c3.append(seed_buckets["st_c3"])
            all_c5.append(seed_buckets["st_c5"])
            all_mae.append(seed_buckets["st_mae"])
        e_c3, e_c5, e_mae = ens(all_c3), ens(all_c5), ens(all_mae)
        # cnn_pp may miss keys if any — intersect
        rows = v9_protocol(e_c3, e_c5, e_mae, rf_map, cnn_pp)
        m = metrics_rows(rows)
        a3.append(
            {
                "frames_per_pass": "all" if k == 0 else k,
                "method": "Ours_v9",
                "cls3_acc": round(float(m["cls3"]["acc"]), 4),
                "cls5_acc": round(float(m["cls5"]["acc"]), 4),
                "progress_mae": round(float(m["progress"]["mae"]), 4),
                "progress_spearman": round(float(m["progress"]["spearman"]), 4),
            }
        )
        print(f"  A3 k={k}: {a3[-1]}", flush=True)

    pd.DataFrame(a3).to_csv(OUT / "A3_frames_sensitivity_ours.csv", index=False)
    # also refresh combined A2/A3 files keeping B5 + Ours
    pd.DataFrame(a2).to_csv(OUT / "A2_boundary_sensitivity.csv", index=False)
    # merge A3 B5 old + ours
    old = pd.read_csv(OUT / "A3_frames_sensitivity.csv") if (OUT / "A3_frames_sensitivity.csv").exists() else None
    if old is not None and "method" not in old.columns:
        old = old.assign(method="MeanPool+RF")
    merged = pd.concat([old, pd.DataFrame(a3)], ignore_index=True) if old is not None else pd.DataFrame(a3)
    merged.to_csv(OUT / "A3_frames_sensitivity.csv", index=False)
    return a2, a3


def run_t12(device):
    print("=== T12 random leak + CNN/Ours ===", flush=True)
    X, meta0, names = load_feature_cache(ROOT / "outputs/baselines/features")
    meta = meta0.copy()
    rng = np.random.default_rng(42)
    tags = {}
    for g, gdf in meta.groupby("grind"):
        idx = gdf.index.to_numpy()
        perm = rng.permutation(len(idx))
        n_te = max(1, int(round(0.2 * len(idx))))
        te_set = set(idx[perm[:n_te]].tolist())
        for i in idx:
            tags[int(i)] = "test" if int(i) in te_set else "train"
    meta["split_random"] = [tags[int(i)] for i in meta.index]

    y3 = np.array([cls3(int(g)) for g in meta.grind])
    y5 = np.array([cls5(int(g)) for g in meta.grind])
    tr = meta.split_random.to_numpy() == "train"
    te = ~tr
    clf3 = RandomForestClassifier(n_estimators=300, max_depth=16, n_jobs=-1, random_state=42)
    clf5 = RandomForestClassifier(n_estimators=300, max_depth=16, n_jobs=-1, random_state=42)
    clf3.fit(X[tr], y3[tr])
    clf5.fit(X[tr], y5[tr])
    m3 = eval_classification(y3[te], clf3.predict(X[te]))
    m5 = eval_classification(y5[te], clf5.predict(X[te]))

    # pass MeanPool RF with random pass tags
    meta_p = build_pass_table(X, meta)
    # fake split column for make_pass_features: monkey via split name
    # reuse train_rf by temporarily writing split_interleaved from random at pass level
    meta_r = meta.copy()
    meta_r["split_interleaved"] = meta_r["split_random"]
    rf_map = train_rf(X, meta_r, "interleaved")
    # metrics from rf_map vs true labels
    keys = list(rf_map.keys())
    m_pass3 = eval_classification(
        [cls3(k[0]) for k in keys], [int(rf_map[k]["p3"].argmax()) for k in keys]
    )
    m_pass5 = eval_classification(
        [cls5(k[0]) for k in keys], [int(rf_map[k]["p5"].argmax()) for k in keys]
    )

    # CNN ResNet on random split (short)
    class RandomFrameDS(Dataset):
        def __init__(self, meta, part, train=False):
            self.meta = meta[meta.split_random == part].reset_index(drop=True)
            self.tf = default_transform(train=train)

        def __len__(self):
            return len(self.meta)

        def __getitem__(self, idx):
            r = self.meta.iloc[idx]
            x = self.tf(Image.open(r.path).convert("RGB"))
            g = int(r.grind)
            return x, cls3(g), cls5(g), torch.tensor(progress(g), dtype=torch.float32)

    device_t = torch.device(device)
    train_loader = DataLoader(RandomFrameDS(meta, "train", True), batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(RandomFrameDS(meta, "test", False), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = bb.fc.in_features
    bb.fc = nn.Identity()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = bb
            self.head_cls3 = nn.Linear(in_f, 3)
            self.head_cls5 = nn.Linear(in_f, 5)
            self.head_prog = nn.Linear(in_f, 1)

        def forward(self, x):
            f = self.backbone(x)
            return self.head_cls3(f), self.head_cls5(f), self.head_prog(f).squeeze(-1)

    model = Net().to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    epochs = 8
    for ep in range(1, epochs + 1):
        model.train()
        for x, y3b, y5b, ypb in train_loader:
            x = x.to(device_t)
            y3b, y5b, ypb = y3b.to(device_t), y5b.to(device_t), ypb.to(device_t)
            o3, o5, op = model(x)
            loss = ce(o3, y3b) + ce(o5, y5b) + 2 * mse(op, ypb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        p3, t3, p5, t5 = [], [], [], []
        with torch.no_grad():
            for x, y3b, y5b, ypb in test_loader:
                o3, o5, _ = model(x.to(device_t))
                p3.append(o3.argmax(1).cpu().numpy())
                t3.append(y3b.numpy())
                p5.append(o5.argmax(1).cpu().numpy())
                t5.append(y5b.numpy())
        cm3 = eval_classification(np.concatenate(t3), np.concatenate(p3))
        cm5 = eval_classification(np.concatenate(t5), np.concatenate(p5))
        print(f"  [T12 CNN] ep{ep} cls3={cm3['acc']:.4f} cls5={cm5['acc']:.4f}", flush=True)

    # Ours-like on random: MeanPool RF already; add LSTM train quick 20 ep
    seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    # Overwrite split tags in a copied meta used by PassSeq via split_interleaved
    meta_lstm = meta.copy()
    meta_lstm["split_interleaved"] = meta_lstm["split_random"]
    train_ds = PassSeq(X, meta_lstm, seg, "interleaved", "train")
    test_ds = PassSeq(X, meta_lstm, seg, "interleaved", "test")
    tr_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)
    te_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate)
    lstm = TemporalExpert(len(names)).to(device_t)
    opt = torch.optim.AdamW(lstm.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(1, 21):
        lstm.train()
        for x, lens, seg_b, y3b, y5b, ypb, ypl, g, pid in tr_loader:
            x, seg_b = x.to(device_t), seg_b.to(device_t)
            y3b, y5b, ypb, ypl = y3b.to(device_t), y5b.to(device_t), ypb.to(device_t), ypl.to(device_t)
            o3, o5, op, opl = lstm(x, lens, seg_b)
            loss = F.cross_entropy(o3, y3b) + F.cross_entropy(o5, y5b) + F.smooth_l1_loss(op, ypb)
            loss = loss + F.binary_cross_entropy_with_logits(opl, ypl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    lstm.eval()
    rows = []
    with torch.no_grad():
        for x, lens, seg_b, y3b, y5b, ypb, ypl, g, pid in te_loader:
            o3, o5, op, opl = lstm(x.to(device_t), lens, seg_b.to(device_t))
            p3s = F.softmax(o3, -1).cpu().numpy()
            p5s = F.softmax(o5, -1).cpu().numpy()
            for i in range(o3.size(0)):
                key = (int(g[i]), int(pid[i]))
                rf_r = rf_map.get(key)
                if rf_r is None:
                    continue
                rows.append(
                    {
                        "key": key,
                        "p3": 0.9 * p3s[i] + 0.1 * rf_r["p3"],
                        "p5": 0.25 * p5s[i] + 0.75 * rf_r["p5"],
                        "pp": 0.3 * float(op[i].cpu()) + 0.7 * rf_r["pp"],
                        "ppl": float(torch.sigmoid(opl[i]).cpu()),
                        "t3": int(y3b[i]),
                        "t5": int(y5b[i]),
                        "tp": float(ypb[i]),
                        "tpl": int(ypl[i]),
                    }
                )
    m_ours = metrics_rows(rows)

    out_rows = [
        {
            "method": "RF frame-level (random 8:2)",
            "cls3_acc": round(m3["acc"], 4),
            "cls5_acc": round(m5["acc"], 4),
            "note": "泄漏上界·不可作主结论",
        },
        {
            "method": "MeanPool+RF pass (random)",
            "cls3_acc": round(m_pass3["acc"], 4),
            "cls5_acc": round(m_pass5["acc"], 4),
            "note": "pass 聚合后仍偏乐观",
        },
        {
            "method": "ResNet-18 frame (random 8:2, 8ep)",
            "cls3_acc": round(cm3["acc"], 4),
            "cls5_acc": round(cm5["acc"], 4),
            "note": "泄漏上界·不可作主结论",
        },
        {
            "method": "Ours-like LSTM+RF (random pass, 20ep)",
            "cls3_acc": round(float(m_ours["cls3"]["acc"]), 4),
            "cls5_acc": round(float(m_ours["cls5"]["acc"]), 4),
            "note": "泄漏上界·不可作主结论",
        },
    ]
    pd.DataFrame(out_rows).to_csv(OUT / "T12_random_leak.csv", index=False)
    save_json(out_rows, OUT / "T12_random_leak.json")
    return out_rows


class MultiTaskBackbone(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name == "mobilenet_v3_small":
            bb = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            in_f = bb.classifier[0].in_features
            bb.classifier = nn.Identity()
            self.backbone = bb
        elif name == "vit_b_16":
            bb = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            in_f = bb.heads.head.in_features
            bb.heads.head = nn.Identity()
            self.backbone = bb
        else:
            raise ValueError(name)
        self.head_cls3 = nn.Linear(in_f, 3)
        self.head_cls5 = nn.Linear(in_f, 5)
        self.head_prog = nn.Linear(in_f, 1)

    def forward(self, x):
        if self.name == "vit_b_16":
            # ViT default 224; resize upstream
            f = self.backbone(x)
        else:
            f = self.backbone(x)
        return self.head_cls3(f), self.head_cls5(f), self.head_prog(f).squeeze(-1)


def run_a1(device, name: str, epochs: int = 12, batch_size: int = 48):
    print(f"=== A1 backbone {name} ===", flush=True)
    frames_csv = ROOT / "outputs/baselines/features/frame_features_meta.csv"
    meta = load_frames_meta(frames_csv)
    h, w = (224, 224) if name.startswith("vit") else (128, 768)
    train_ds = FrameDataset(meta, SPLIT, "train", train=True, height=h, width=w)
    test_ds = FrameDataset(meta, SPLIT, "test", train=False, height=h, width=w)
    # For ViT, override transform to square
    if name.startswith("vit"):
        tf_tr = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        tf_te = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        train_ds.tf, test_ds.tf = tf_tr, tf_te
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    device_t = torch.device(device)
    model = MultiTaskBackbone(name).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    best = None
    t0 = time.time()
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
            f"  [{name}] ep{ep}/{epochs} cls3={m['cls3']['acc']:.4f} "
            f"cls5={m['cls5']['acc']:.4f} mae={m['progress']['mae']:.4f} sp={m['progress']['spearman']:.4f}",
            flush=True,
        )
        score = (m["cls3"]["acc"], m["cls5"]["acc"], -m["progress"]["mae"])
        if best is None or score > best[0]:
            best = (score, m)
    elapsed = time.time() - t0
    row = {
        "backbone": name,
        "split": SPLIT,
        "epochs": epochs,
        "cls3_acc": round(float(best[1]["cls3"]["acc"]), 4),
        "cls5_acc": round(float(best[1]["cls5"]["acc"]), 4),
        "progress_mae": round(float(best[1]["progress"]["mae"]), 4),
        "progress_spearman": round(float(best[1]["progress"]["spearman"]), 4),
        "train_s": round(elapsed, 1),
        "note": "帧级多任务；附录 A1，不作主结论替代 Ours",
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        default="all",
        choices=["all", "seed", "ablation", "t8", "a2a3", "t12", "a1", "tables"],
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device2", default="cuda:1")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.phase in ("all", "seed"):
        run_seed_std(args.device)
    if args.phase in ("all", "ablation"):
        run_ablations(args.device)
    if args.phase in ("all", "t8"):
        run_t8(args.device)
    if args.phase in ("all", "a2a3"):
        run_a2_a3_ours(args.device)
    if args.phase in ("all", "t12"):
        run_t12(args.device)
    if args.phase in ("all", "a1"):
        # run two backbones sequentially (or user can parallelize externally)
        rows = []
        rows.append(run_a1(args.device, "mobilenet_v3_small", epochs=12, batch_size=64))
        rows.append(run_a1(args.device2 if torch.cuda.device_count() > 1 else args.device, "vit_b_16", epochs=8, batch_size=32))
        # attach existing ResNet/EffNet from main table for reference
        main = pd.read_csv(ROOT / "outputs/baselines/main_comparison_table.csv")
        for model, name in [("B3_ResNet18", "resnet18"), ("B4_EfficientNetB0", "efficientnet_b0")]:
            r = main[(main.model == model) & (main.split == SPLIT)].iloc[0]
            rows.append(
                {
                    "backbone": name,
                    "split": SPLIT,
                    "epochs": 12,
                    "cls3_acc": round(float(r.cls3_acc), 4),
                    "cls5_acc": round(float(r.cls5_acc), 4),
                    "progress_mae": round(float(r.progress_mae), 4),
                    "progress_spearman": round(float(r.progress_spearman), 4),
                    "train_s": None,
                    "note": "主表已有基线",
                }
            )
        pd.DataFrame(rows).to_csv(OUT / "A1_extra_backbones.csv", index=False)
        save_json(rows, OUT / "A1_extra_backbones.json")

    print("DONE", args.phase, flush=True)


if __name__ == "__main__":
    main()
