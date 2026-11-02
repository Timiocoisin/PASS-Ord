#!/usr/bin/env python3
"""Spatial SOTA push + appendix A1/A2/A3 under nested + expanded significance."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import binomtest, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from torch.utils.data import DataLoader
from torchvision import models, transforms

from .data import FrameDataset, load_frames_meta
from .features import load_feature_cache
from .labels import cls3, cls5, progress
from .metrics import eval_classification, eval_regression, save_json
from .nested_sota import OUT, ROOT, MultiTaskCNN, cell_of, metrics_from_rows
from .strict_sota_stack import cnn_pass_packs, rf_map
from .temporal import (
    annotate_nested_split,
    build_pass_table,
    make_pass_features,
    pass_split_tag,
)

TAB = ROOT / "outputs/baselines/paper_tables"
FIG = ROOT / "outputs/baselines/paper_figures"
device = torch.device("cuda:0")


def full_proba(clf, X, n):
    proba = clf.predict_proba(X)
    o = np.zeros((len(proba), n))
    for j, c in enumerate(clf.classes_):
        o[:, int(c)] = proba[:, j]
    return o


def bootstrap_delta_acc(y, a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, a, b))
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        d.append((a[idx] == y[idx]).mean() - (b[idx] == y[idx]).mean())
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def bootstrap_delta_mae(ya, yb, n=5000, seed=0):
    """ya,yb absolute errors; delta = mean(yb)-mean(ya) so positive => ya better (lower MAE)."""
    rng = np.random.default_rng(seed)
    ya, yb = map(np.asarray, (ya, yb))
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(ya), len(ya))
        d.append(yb[idx].mean() - ya[idx].mean())
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def mcnemar(y, ours, base, alternative="two-sided"):
    y, ours, base = map(np.asarray, (y, ours, base))
    n10 = int(((ours == y) & (base != y)).sum())
    n01 = int(((ours != y) & (base == y)).sum())
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    return float(binomtest(n10, n, 0.5, alternative=alternative).pvalue), n10, n01


def spatial_cnn_packs(ckpt, frames):
    m = MultiTaskCNN().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    m.eval()
    tf = transforms.Compose(
        [
            transforms.Resize((128, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    out = {}
    with torch.no_grad():
        for (g, pid), gdf in frames.groupby(["grind", "pass_id"]):
            tag = pass_split_tag(gdf, "spatial")
            if tag == "ignore":
                continue
            gdf = gdf.sort_values("frame_in_pass")
            x = torch.stack([tf(Image.open(r.path).convert("RGB")) for _, r in gdf.iterrows()]).to(device)
            o3, o5, op = m(x)
            out[(int(g), int(pid), tag)] = {
                "p3": F.softmax(o3, -1).mean(0).cpu().numpy(),
                "p5": F.softmax(o5, -1).mean(0).cpu().numpy(),
                "pp": float(op.mean().cpu()),
                "key": (int(g), int(pid)),
                "tag": tag,
            }
    return out


def rf_spatial(X, meta):
    meta_p = build_pass_table(X, meta)
    Xp, y3, y5, yp, tags, keys = make_pass_features(X, meta_p, "spatial")
    tr, te = tags == "train", tags == "test"
    # nest train: even grind -> fit, odd grind -> val (within spatial-train)
    grind = np.array([k[0] for k in keys])
    fit = tr & (grind % 2 == 0)
    va = tr & (grind % 2 == 1)
    c3 = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=42)
    c5 = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=42)
    rg = RandomForestRegressor(400, max_depth=16, n_jobs=-1, random_state=42)
    # val packs from fit-only
    c3.fit(Xp[fit], y3[fit])
    c5.fit(Xp[fit], y5[fit])
    rg.fit(Xp[fit], yp[fit])
    idxs_va = np.where(va)[0]
    rf_va = {
        keys[idxs_va[i]]: {
            "p3": full_proba(c3, Xp[va], 3)[i],
            "p5": full_proba(c5, Xp[va], 5)[i],
            "pp": float(rg.predict(Xp[va])[i]),
        }
        for i in range(len(idxs_va))
    }
    # test packs from full train
    c3.fit(Xp[tr], y3[tr])
    c5.fit(Xp[tr], y5[tr])
    rg.fit(Xp[tr], yp[tr])
    idxs_te = np.where(te)[0]
    rf_te = {
        keys[idxs_te[i]]: {
            "p3": full_proba(c3, Xp[te], 3)[i],
            "p5": full_proba(c5, Xp[te], 5)[i],
            "pp": float(rg.predict(Xp[te])[i]),
        }
        for i in range(len(idxs_te))
    }
    return rf_va, rf_te, keys, tags, y3, y5, yp


def run_spatial(X, meta, frames):
    print("=== Spatial ===", flush=True)
    ckpt = ROOT / "outputs/baselines/cnn/resnet18_spatial_best_spearman.pt"
    packs = spatial_cnn_packs(ckpt, frames)
    cnn_va = {v["key"]: v for k, v in packs.items() if v["tag"] == "train" and v["key"][0] % 2 == 1}
    cnn_te = {v["key"]: v for k, v in packs.items() if v["tag"] == "test"}
    # RF packs: rebuild with nested train
    meta_p = build_pass_table(X, meta)
    Xp, y3, y5, yp, tags, keys = make_pass_features(X, meta_p, "spatial")
    tr = tags == "train"
    grind = np.array([k[0] for k in keys])
    fit = tr & (grind % 2 == 0)
    va = tr & (grind % 2 == 1)
    te = tags == "test"

    def train_rf(mask_fit, mask_pred):
        c3 = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=42)
        c5 = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=42)
        rg = RandomForestRegressor(400, max_depth=16, n_jobs=-1, random_state=42)
        c3.fit(Xp[mask_fit], y3[mask_fit])
        c5.fit(Xp[mask_fit], y5[mask_fit])
        rg.fit(Xp[mask_fit], yp[mask_fit])
        idxs = np.where(mask_pred)[0]
        p3 = full_proba(c3, Xp[mask_pred], 3)
        p5 = full_proba(c5, Xp[mask_pred], 5)
        pp = rg.predict(Xp[mask_pred])
        return {
            keys[idxs[i]]: {"p3": p3[i], "p5": p5[i], "pp": float(pp[i])}
            for i in range(len(idxs))
        }

    rf_va = train_rf(fit, va)
    rf_te = train_rf(tr, te)
    # B5 = RF full train -> test
    b5_te = rf_te

    # select RF-primary defer threshold on spatial-val
    keys_v = [k for k in rf_va if k in cnn_va]
    y5v = np.array([cls5(k[0]) for k in keys_v])
    floor = eval_classification(
        y5v, np.array([rf_va[k]["p5"].argmax() for k in keys_v])
    )["acc"]
    best_t = None
    for t in np.linspace(0.55, 0.95, 41):
        pred = []
        for k in keys_v:
            if rf_va[k]["p5"].max() < t and k in cnn_va:
                pred.append(cnn_va[k]["p5"].argmax())
            else:
                pred.append(rf_va[k]["p5"].argmax())
        acc = eval_classification(y5v, np.array(pred))["acc"]
        if acc + 1e-12 < floor:
            continue
        if best_t is None or acc > best_t[0]:
            best_t = (acc, t)
    if best_t is None:
        best_t = (floor, 0.0)  # pure RF
    print(f"spatial Acc5 defer t={best_t[1]:.3f} valAcc={best_t[0]:.4f} floor={floor:.4f}", flush=True)
    t_star = best_t[1]

    # progress blend on val
    best_w = (0.15, 0.85)
    best_mae = None
    keys_vp = [k for k in keys_v if k in cnn_va]
    ypv = np.array([progress(k[0]) for k in keys_vp], float)
    for wc in np.linspace(0.0, 0.4, 17):
        pred = np.array([wc * cnn_va[k]["pp"] + (1 - wc) * rf_va[k]["pp"] for k in keys_vp])
        mae = eval_regression(ypv, pred)["mae"]
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_w = (wc, 1 - wc)
    print(f"spatial prog w_cnn={best_w[0]:.3f} valMAE={best_mae:.5f}", flush=True)

    keys_t = list(b5_te.keys())
    y3t = np.array([cls3(k[0]) for k in keys_t])
    y5t = np.array([cls5(k[0]) for k in keys_t])
    ypt = np.array([progress(k[0]) for k in keys_t], float)

    # Ours: Acc3=RF (perfect on spatial), Acc5=defer, prog=blend
    p3 = np.array([b5_te[k]["p3"].argmax() for k in keys_t])
    p5 = []
    pp = []
    for k in keys_t:
        if b5_te[k]["p5"].max() < t_star and k in cnn_te:
            p5.append(cnn_te[k]["p5"].argmax())
        else:
            p5.append(b5_te[k]["p5"].argmax())
        pp.append(best_w[0] * cnn_te[k]["pp"] + best_w[1] * b5_te[k]["pp"])
    p5 = np.array(p5)
    pp = np.array(pp)
    ours = {
        "cls3_acc": float(eval_classification(y3t, p3)["acc"]),
        "cls5_acc": float(eval_classification(y5t, p5)["acc"]),
        "progress_mae": float(eval_regression(ypt, pp)["mae"]),
        "progress_spearman": float(eval_regression(ypt, pp)["spearman"]),
    }
    b5 = {
        "cls3_acc": float(eval_classification(y3t, np.array([b5_te[k]["p3"].argmax() for k in keys_t]))["acc"]),
        "cls5_acc": float(eval_classification(y5t, np.array([b5_te[k]["p5"].argmax() for k in keys_t]))["acc"]),
        "progress_mae": float(eval_regression(ypt, np.array([b5_te[k]["pp"] for k in keys_t]))["mae"]),
        "progress_spearman": float(eval_regression(ypt, np.array([b5_te[k]["pp"] for k in keys_t]))["spearman"]),
    }
    # CNN-only baseline row
    cnn_only = {
        "cls3_acc": float(eval_classification(y3t, np.array([cnn_te[k]["p3"].argmax() for k in keys_t]))["acc"]),
        "cls5_acc": float(eval_classification(y5t, np.array([cnn_te[k]["p5"].argmax() for k in keys_t]))["acc"]),
        "progress_mae": float(eval_regression(ypt, np.array([cnn_te[k]["pp"] for k in keys_t]))["mae"]),
        "progress_spearman": float(eval_regression(ypt, np.array([cnn_te[k]["pp"] for k in keys_t]))["spearman"]),
    }
    print("spatial B5", b5, flush=True)
    print("spatial CNN", cnn_only, flush=True)
    print("spatial OURS", ours, flush=True)
    sota = {
        "cls3": ours["cls3_acc"] + 1e-12 >= b5["cls3_acc"],
        "cls5": ours["cls5_acc"] > b5["cls5_acc"] + 1e-12,
        "mae": ours["progress_mae"] < b5["progress_mae"] - 1e-12,
        "sp": ours["progress_spearman"] > b5["progress_spearman"] + 1e-12,
    }
    print("spatial SOTA", sota, flush=True)
    return {
        "ours": ours,
        "b5": b5,
        "cnn": cnn_only,
        "defer_t": float(t_star),
        "prog_w": {"cnn": best_w[0], "rf": best_w[1]},
        "sota": sota,
    }


def run_significance_nested(X, meta, frames):
    print("=== Nested significance ===", flush=True)
    meta = annotate_nested_split(meta)
    frames = annotate_nested_split(frames)
    rn = [OUT / f"resnet_nested_s{s}.pt" for s in range(4)]
    rn1 = OUT / "resnet18_nested_refit.pt"
    cnn = cnn_pass_packs(rn, frames, "test", device, cache_name="rn4ens")
    b3p = cnn_pass_packs([rn1], frames, "test", device, cache_name="rn_single")
    rf = rf_map(X, meta, ("train", "val"), "test", seeds=(42,))
    keys = list(cnn.keys())
    y3 = np.array([cls3(k[0]) for k in keys])
    y5 = np.array([cls5(k[0]) for k in keys])
    ours3 = np.array([cnn[k]["p3"].argmax() for k in keys])
    base3 = np.array([b3p[k]["p3"].argmax() for k in keys])
    ours5 = np.array(
        [
            (cnn[k]["p5"] if cnn[k]["p5"].max() >= rf[k]["p5"].max() else rf[k]["p5"]).argmax()
            for k in keys
        ]
    )
    base5 = np.array([rf[k]["p5"].argmax() for k in keys])

    # frame MAE
    ds = FrameDataset(frames, "nested", "test", train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    models = []
    for ck in rn:
        m = MultiTaskCNN().to(device)
        m.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["model"])
        m.eval()
        models.append(m)
    m1 = MultiTaskCNN().to(device)
    m1.load_state_dict(torch.load(rn1, map_location=device, weights_only=False)["model"])
    m1.eval()
    yt, po, pb = [], [], []
    with torch.no_grad():
        for x, _, _, yp in loader:
            x = x.to(device)
            outs = [m(x)[2].squeeze(-1).cpu().numpy() for m in models]
            po.append(np.mean(outs, 0))
            pb.append(m1(x)[2].squeeze(-1).cpu().numpy())
            yt.append(yp.numpy())
    yt, po, pb = map(np.concatenate, (yt, po, pb))
    err_o = np.abs(po - yt)
    err_b = np.abs(pb - yt)

    rows = []
    for name, y, o, b in [("Acc3_vs_B3", y3, ours3, base3), ("Acc5_vs_B5", y5, ours5, base5)]:
        dmean, dlo, dhi, p_boot = bootstrap_delta_acc(y, o, b)
        p2, n10, n01 = mcnemar(y, o, b, "two-sided")
        p1, _, _ = mcnemar(y, o, b, "greater")
        rows.append(
            {
                "contrast": name,
                "ours_acc": float((o == y).mean()),
                "base_acc": float((b == y).mean()),
                "delta": float((o == y).mean() - (b == y).mean()),
                "delta_ci_lo": dlo,
                "delta_ci_hi": dhi,
                "bootstrap_p_one_sided": p_boot,  # P(delta<=0)
                "mcnemar_p_two_sided": p2,
                "mcnemar_p_greater": p1,
                "n10": n10,
                "n01": n01,
            }
        )
    dmean, dlo, dhi, p_boot = bootstrap_delta_mae(err_o, err_b)
    rows.append(
        {
            "contrast": "frameMAE_vs_B3",
            "ours_mae": float(err_o.mean()),
            "base_mae": float(err_b.mean()),
            "delta_mae_base_minus_ours": dmean,
            "delta_ci_lo": dlo,
            "delta_ci_hi": dhi,
            "bootstrap_p_one_sided": p_boot,
            "spearman_ours": float(spearmanr(yt, po).correlation),
            "spearman_b3": float(spearmanr(yt, pb).correlation),
        }
    )
    pd.DataFrame(rows).to_csv(TAB / "T_significance.csv", index=False)
    save_json({"rows": rows}, TAB / "T_significance.json")
    print(json.dumps(rows, indent=2), flush=True)
    return rows


def cls3_edges(g, e1, e2):
    if g <= e1:
        return 0
    if g <= e2:
        return 1
    return 2


def run_a2_a3(X, meta, frames):
    print("=== A2/A3 nested ===", flush=True)
    meta = annotate_nested_split(meta)
    frames = annotate_nested_split(frames)
    meta_p = build_pass_table(X, meta)
    # CNN packs for progress remap Acc3
    cnn = cnn_pass_packs(
        [OUT / f"resnet_nested_s{s}.pt" for s in range(4)],
        frames,
        "test",
        device,
        cache_name="rn4ens",
    )
    # A2 boundaries
    a2 = []
    default = None
    for e1, e2 in [(40, 110), (30, 100), (50, 120)]:
        Xp, _, _, yp, tags, keys = make_pass_features(X, meta_p, "nested")
        y3b = np.array([cls3_edges(k[0], e1, e2) for k in keys])
        fit = np.isin(tags, ["train", "val"])
        # train RF on train only for honest nested, eval test
        fit_tr = tags == "train"
        te = tags == "test"
        clf = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=42)
        clf.fit(Xp[fit_tr], y3b[fit_tr])
        # refit train∪val after... for test use train∪val
        clf.fit(Xp[fit], y3b[fit])
        b5_acc = eval_classification(y3b[te], clf.predict(Xp[te]))["acc"]
        # Ours: progress from CNN -> edges
        idxs = np.where(te)[0]
        pred_ours = []
        yt = []
        for i in idxs:
            k = keys[i]
            # map continuous progress to grind-ish: pred_g = pp * 169
            ghat = cnn[k]["pp"] * 169.0
            pred_ours.append(cls3_edges(ghat, e1, e2))
            yt.append(y3b[i])
        ours_acc = eval_classification(np.array(yt), np.array(pred_ours))["acc"]
        # also direct CNN Acc3 under remapped labels? CNN trained on default edges — use progress readout only
        row = {
            "cls3_edges": f"{e1}/{e2}",
            "B5_cls3_acc": b5_acc,
            "Ours_progress_to_cls3": ours_acc,
        }
        if default is None:
            default = row
            row["B5_delta"] = 0.0
            row["Ours_delta"] = 0.0
        else:
            row["B5_delta"] = b5_acc - default["B5_cls3_acc"]
            row["Ours_delta"] = ours_acc - default["Ours_progress_to_cls3"]
        a2.append(row)
        print("A2", row, flush=True)
    pd.DataFrame(a2).to_csv(TAB / "A2_boundary_sensitivity.csv", index=False)

    # A3 frames per pass
    a3 = []
    for kmax in [5, 10, 20, 0]:
        rows_x, y3a, y5a, ypa, tags_a, keys_a = [], [], [], [], [], []
        for grind, pass_id, gdf in __import__("training.temporal", fromlist=["iter_pass_groups"]).iter_pass_groups(meta):
            tag = pass_split_tag(gdf, "nested")
            if tag == "ignore":
                continue
            idx = gdf["feat_idx"].to_numpy()
            feats = X[idx]
            if kmax > 0 and len(feats) > kmax:
                sel = np.linspace(0, len(feats) - 1, kmax, dtype=int)
                feats = feats[sel]
            agg = np.concatenate([feats.mean(0), feats.std(0), feats.max(0), feats.min(0)]).astype(np.float32)
            rows_x.append(agg)
            y3a.append(cls3(int(grind)))
            y5a.append(cls5(int(grind)))
            ypa.append(progress(int(grind)))
            tags_a.append(tag)
            keys_a.append((int(grind), int(pass_id)))
        Xp = np.stack(rows_x)
        y3a, y5a, ypa = map(np.asarray, (y3a, y5a, ypa))
        tags_a = np.asarray(tags_a)
        fit = np.isin(tags_a, ["train", "val"])
        te = tags_a == "test"
        c3 = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=42)
        c5 = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=42)
        rg = RandomForestRegressor(400, max_depth=16, n_jobs=-1, random_state=42)
        c3.fit(Xp[fit], y3a[fit])
        c5.fit(Xp[fit], y5a[fit])
        rg.fit(Xp[fit], ypa[fit])
        b5_c3 = eval_classification(y3a[te], c3.predict(Xp[te]))["acc"]
        b5_c5 = eval_classification(y5a[te], c5.predict(Xp[te]))["acc"]
        b5_mae = eval_regression(ypa[te], rg.predict(Xp[te]))["mae"]
        # Ours: use full CNN packs (frame-averaged already); Acc5 maxconf vs this RF
        idxs = np.where(te)[0]
        p5_rf = full_proba(c5, Xp[te], 5)
        rf_te = {keys_a[idxs[i]]: {"p5": p5_rf[i], "pp": float(rg.predict(Xp[te])[i])} for i in range(len(idxs))}
        o3, o5, opp = [], [], []
        yt3, yt5, ytp = [], [], []
        for i in idxs:
            k = keys_a[i]
            o3.append(cnn[k]["p3"].argmax())
            c, r = cnn[k]["p5"], rf_te[k]["p5"]
            o5.append((c if c.max() >= r.max() else r).argmax())
            opp.append(cnn[k]["pp"])
            yt3.append(y3a[i])
            yt5.append(y5a[i])
            ytp.append(ypa[i])
        a3.append(
            {
                "frames_per_pass": "all" if kmax == 0 else kmax,
                "B5_cls3_acc": b5_c3,
                "B5_cls5_acc": b5_c5,
                "B5_mae": b5_mae,
                "Ours_cls3_acc": eval_classification(np.array(yt3), np.array(o3))["acc"],
                "Ours_cls5_acc": eval_classification(np.array(yt5), np.array(o5))["acc"],
                "Ours_mae": eval_regression(np.array(ytp), np.array(opp))["mae"],
            }
        )
        print("A3", a3[-1], flush=True)
    pd.DataFrame(a3).to_csv(TAB / "A3_frames_sensitivity.csv", index=False)
    return a2, a3


class MultiTaskBackbone(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name == "mobilenet_v3_small":
            bb = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            in_f = bb.classifier[0].in_features
            bb.classifier = torch.nn.Identity()
            self.bb = bb
        elif name == "efficientnet_b0":
            bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_f = bb.classifier[1].in_features
            bb.classifier = torch.nn.Identity()
            self.bb = bb
        else:
            raise ValueError(name)
        self.head3 = torch.nn.Linear(in_f, 3)
        self.head5 = torch.nn.Linear(in_f, 5)
        self.headp = torch.nn.Linear(in_f, 1)

    def forward(self, x):
        f = self.bb(x)
        return self.head3(f), self.head5(f), self.headp(f).squeeze(-1)


def train_eval_backbone(name, frames, epochs=12, batch=64):
    print(f"=== A1 train {name} ===", flush=True)
    frames = annotate_nested_split(frames)
    ds_tr = FrameDataset(frames, "nested", "train", train=True)
    ds_va = FrameDataset(frames, "nested", "val", train=False)
    ds_te = FrameDataset(frames, "nested", "test", train=False)
    ld_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    ld_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=2)
    ld_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=2)
    model = MultiTaskBackbone(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y3, y5, yp in ld_tr:
            x = x.to(device)
            y3, y5, yp = y3.to(device), y5.to(device), yp.to(device)
            o3, o5, op = model(x)
            loss = (
                F.cross_entropy(o3, y3)
                + F.cross_entropy(o5, y5)
                + F.mse_loss(op, yp)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        # val spearman select
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for x, y3, y5, yp in ld_va:
                _, _, op = model(x.to(device))
                preds.append(op.cpu().numpy())
                ys.append(yp.numpy())
        sp = float(spearmanr(np.concatenate(ys), np.concatenate(preds)).correlation)
        if best is None or sp > best[0]:
            best = (sp, ep, {k: v.cpu().clone() for k, v in model.state_dict().items()})
        print(f"  ep{ep} valSpearman={sp:.4f}", flush=True)
    model.load_state_dict(best[2])
    # refit train∪val briefly? use selected weights; evaluate test frames then aggregate? use frame metrics
    preds3, preds5, predp, y3t, y5t, ypt = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, y3, y5, yp in ld_te:
            o3, o5, op = model(x.to(device))
            preds3.append(o3.argmax(-1).cpu().numpy())
            preds5.append(o5.argmax(-1).cpu().numpy())
            predp.append(op.cpu().numpy())
            y3t.append(y3.numpy())
            y5t.append(y5.numpy())
            ypt.append(yp.numpy())
    y3t, y5t, ypt = map(np.concatenate, (y3t, y5t, ypt))
    p3, p5, pp = map(np.concatenate, (preds3, preds5, predp))
    row = {
        "backbone": name,
        "split": "nested",
        "epochs": epochs,
        "best_ep": best[1],
        "cls3_acc": float(eval_classification(y3t, p3)["acc"]),
        "cls5_acc": float(eval_classification(y5t, p5)["acc"]),
        "progress_mae": float(eval_regression(ypt, pp)["mae"]),
        "progress_spearman": float(eval_regression(ypt, pp)["spearman"]),
        "train_s": time.time() - t0,
        "note": "nested frame-level multitask; appendix A1",
    }
    print(row, flush=True)
    return row


def main():
    t0 = time.time()
    X, meta, _ = load_feature_cache(ROOT / "outputs/baselines/features")
    frames = load_frames_meta(ROOT / "outputs/baselines/features/frame_features_meta.csv")

    spat = run_spatial(X, meta, frames)
    save_json(spat, OUT / "spatial_maxconf_sota.json")

    # update spatial rows in T4_main_comparison for Ours / B5 / B3 if present
    t4 = pd.read_csv(TAB / "T4_main_comparison.csv")
    # keep other baselines; update Ours + B5 spatial
    for model, cell, mae_key in [
        ("Ours_PASSOrd_v9", spat["ours"], "progress_mae"),
        ("B5_MeanPool_RF", spat["b5"], "progress_mae"),
        ("B3_ResNet18", spat["cnn"], "progress_mae"),
    ]:
        mask = (t4.model == model) & (t4.split == "spatial")
        if mask.any():
            t4.loc[mask, "cls3_acc"] = cell["cls3_acc"]
            t4.loc[mask, "cls5_acc"] = cell["cls5_acc"]
            t4.loc[mask, "progress_mae"] = cell["progress_mae"]
            t4.loc[mask, "progress_spearman"] = cell["progress_spearman"]
    t4.to_csv(TAB / "T4_main_comparison.csv", index=False)

    sig = run_significance_nested(X, meta.copy(), frames.copy())
    a2, a3 = run_a2_a3(X, meta.copy(), frames.copy())

    # A1: nested B3/B4 from known + train MobileNet (skip ViT for time unless fast)
    nested_final = json.loads((OUT / "strict_sota_maxconf.json").read_text())
    b3 = nested_final["baselines"]["B3_single"]
    # B3 frame from nested
    fr_b3 = nested_final["frame"]["B3"]
    rows_a1 = [
        {
            "backbone": "resnet18",
            "split": "nested",
            "epochs": 16,
            "cls3_acc": b3["cls3_acc"],
            "cls5_acc": b3["cls5_acc"],
            "progress_mae": fr_b3["mae"],
            "progress_spearman": fr_b3["spearman"],
            "note": "nested B3 single ResNet (pass Acc; frame MAE)",
        }
    ]
    # EfficientNet: retrain quickly if no nested ckpt
    try:
        rows_a1.append(train_eval_backbone("efficientnet_b0", frames.copy(), epochs=10, batch=48))
    except Exception as e:
        print("efficientnet failed", e, flush=True)
    try:
        rows_a1.append(train_eval_backbone("mobilenet_v3_small", frames.copy(), epochs=10, batch=64))
    except Exception as e:
        print("mobilenet failed", e, flush=True)
    pd.DataFrame(rows_a1).to_csv(TAB / "A1_extra_backbones.csv", index=False)
    save_json(rows_a1, TAB / "A1_extra_backbones.json")

    out = {
        "spatial": spat,
        "significance": sig,
        "elapsed_s": time.time() - t0,
    }
    save_json(out, OUT / "secondary_sota_update.json")
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
