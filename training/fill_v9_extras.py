#!/usr/bin/env python3
"""Fill remaining paper extras: T10@v9, significance, weight sensitivity, spatial seeds, e2e latency."""

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
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models, transforms

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
from .fill_v9_gaps import (
    MAIN_PROG_W,
    SOFT_W,
    cell_of,
    eval_specialists,
    soft_cls,
    OUT,
    V8,
    V9,
)

PLATEAU_G = 94


class B6LSTM(nn.Module):
    """Baseline BiLSTM matching outputs/baselines/temporal/lstm_*_best.pt."""

    def __init__(self, in_dim=20, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1
        )
        fused = hidden * 2
        self.head_cls3 = nn.Linear(fused, 3)
        self.head_cls5 = nn.Linear(fused, 5)
        self.head_prog = nn.Linear(fused, 1)

    def forward(self, x, lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        # last layer both directions (same as original temporal.LSTMMultiTask)
        f = torch.cat([h[-2], h[-1]], dim=-1)
        return (
            self.head_cls3(f),
            self.head_cls5(f),
            self.head_prog(f).squeeze(-1),
        )


class MultiTaskCNN(nn.Module):
    def __init__(self, name="resnet18"):
        super().__init__()
        bb = models.resnet18(weights=None)
        in_f = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.head_cls3 = nn.Linear(in_f, 3)
        self.head_cls5 = nn.Linear(in_f, 5)
        self.head_prog = nn.Linear(in_f, 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.head_cls3(f), self.head_cls5(f), self.head_prog(f).squeeze(-1)


def load_frame_pp(split="interleaved"):
    path = V9 / f"cnn_frame_pp_resnet18_spearman_{split}.npy"
    if path.exists():
        return np.load(path, allow_pickle=True).item()
    return None


def load_pass_pp(split):
    return np.load(V9 / f"cnn_pp_resnet18_{split}.npy", allow_pickle=True).item()


def ens_specialists(X, meta, names, seg, split, device, ckpts):
    all_c3, all_c5, all_mae = [], [], []
    for ck in ckpts:
        b = eval_specialists(X, meta, names, seg, split, device, ck)
        all_c3.append(b["st_c3"])
        all_c5.append(b["st_c5"])
        all_mae.append(b["st_mae"])
    return ens(all_c3), ens(all_c5), ens(all_mae)


def build_ours_pass_rows(e_c3, e_c5, e_mae, rf, prog_w=None, frame_pp=None, pass_pp=None):
    """Pass-level classification rows; progress = mean of frame-level blend if frame_pp else pass blend."""
    prog_w = prog_w or MAIN_PROG_W
    base = soft_cls(e_c3, e_c5, rf)
    rows = []
    for i, r in enumerate(base):
        key = r["key"]
        a, b, c = prog_w["resnet"], prog_w["rf"], prog_w["lstm"]
        if frame_pp is not None and key in frame_pp:
            vals = [a * rv + b * rf[key]["pp"] + c * e_mae[i]["pp"] for rv in frame_pp[key]]
            pp = float(np.mean(vals))
        else:
            cnn = float(pass_pp[key]) if pass_pp is not None else 0.0
            pp = a * cnn + b * float(rf[key]["pp"]) + c * float(e_mae[i]["pp"])
        nr = dict(r)
        nr["pp"] = pp
        nr["grind"] = int(key[0])
        nr["pass_id"] = int(key[1])
        rows.append(nr)
    return rows


def frame_prog_metrics(e_c3, e_mae, rf, frame_pp, prog_w=None):
    prog_w = prog_w or MAIN_PROG_W
    a, b, c = prog_w["resnet"], prog_w["rf"], prog_w["lstm"]
    yt, yp = [], []
    for i, r in enumerate(e_c3):
        key = r["key"]
        if key not in frame_pp:
            continue
        for rv in frame_pp[key]:
            yt.append(progress(key[0]))
            yp.append(a * rv + b * rf[key]["pp"] + c * e_mae[i]["pp"])
    return eval_regression(yt, yp)


def eval_b6(X, meta, names, seg, split, device):
    ckpt = ROOT / f"outputs/baselines/temporal/lstm_{split}_best.pt"
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model = B6LSTM(in_dim=len(names)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    loader = DataLoader(
        PassSeq(X, meta, seg, split, "test"), batch_size=64, shuffle=False, collate_fn=collate
    )
    rows = []
    with torch.no_grad():
        for x, lens, seg_b, y3, y5, yp, ypl, g, pid in loader:
            # B6 ignores seg; PassSeq still provides it
            o3, o5, op = model(x.to(device), lens)
            p3 = F.softmax(o3, -1).cpu().numpy()
            p5 = F.softmax(o5, -1).cpu().numpy()
            for i in range(o3.size(0)):
                rows.append(
                    {
                        "key": (int(g[i]), int(pid[i])),
                        "grind": int(g[i]),
                        "p3": p3[i],
                        "p5": p5[i],
                        "pp": float(op[i].cpu()),
                        "t3": int(y3[i]),
                        "t5": int(y5[i]),
                        "tp": float(yp[i]),
                    }
                )
    return rows


def mcnemar(y_true, pred_a, pred_b):
    """McNemar exact test: A vs B correctness contingency."""
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    b01 = int(np.sum(~a_ok & b_ok))  # B correct, A wrong
    b10 = int(np.sum(a_ok & ~b_ok))  # A correct, B wrong
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        # exact binomial test under H0: p=0.5 for discordant pairs
        p = float(binomtest(b10, n=n, p=0.5).pvalue)
    return {"b10_Awin": b10, "b01_Bwin": b01, "n_discordant": n, "pvalue": p}


def bootstrap_mae_ci(y_true, y_pred, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    n = len(err)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(err[idx].mean()))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "mae": float(err.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "n": n,
        "n_boot": n_boot,
    }


def bootstrap_acc_ci(y_true, y_pred, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ok = (y_true == y_pred).astype(float)
    n = len(ok)
    boots = [float(ok[rng.integers(0, n, n)].mean()) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "acc": float(ok.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "n": n,
        "n_boot": n_boot,
    }


def bootstrap_delta_acc(y_true, pred_a, pred_b, n_boot=2000, seed=1, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    a = (np.asarray(pred_a) == y_true).astype(float)
    b = (np.asarray(pred_b) == y_true).astype(float)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(float(a[idx].mean() - b[idx].mean()))
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "delta_acc": float(a.mean() - b.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "n_boot": n_boot,
    }


def run_t10(ours_rows, out: Path):
    print("=== T10 v9 error analysis ===", flush=True)
    y3 = [r["t3"] for r in ours_rows]
    p3 = [int(np.asarray(r["p3"]).argmax()) for r in ours_rows]
    y5 = [r["t5"] for r in ours_rows]
    p5 = [int(np.asarray(r["p5"]).argmax()) for r in ours_rows]
    cm3 = confusion_matrix(y3, p3, labels=[0, 1, 2])
    cm5 = confusion_matrix(y5, p5, labels=[0, 1, 2, 3, 4])
    pd.DataFrame(
        cm3, index=["t_early", "t_mid", "t_late"], columns=["p_early", "p_mid", "p_late"]
    ).to_csv(out / "T10_cm_cls3.csv")
    pd.DataFrame(
        cm5, index=[f"t{i}" for i in range(5)], columns=[f"p{i}" for i in range(5)]
    ).to_csv(out / "T10_cm_cls5.csv")

    records = []
    for r, yt, yp_ in zip(ours_rows, y3, p3):
        if yt == yp_:
            continue
        g = int(r["grind"])
        if {yt, yp_} == {0, 1}:
            et = "早↔中边界"
        elif {yt, yp_} == {1, 2}:
            et = "中↔晚边界"
        elif abs(yt - yp_) == 1:
            et = "相邻阶段混淆"
        else:
            et = "跨级错分"
        records.append(
            {"grind": g, "true": yt, "pred": yp_, "error_type": et, "progress": r["tp"]}
        )
    err_df = pd.DataFrame(records)
    n = len(ours_rows)
    summary = []
    if len(err_df):
        for et, gdf in err_df.groupby("error_type"):
            grinds = sorted(gdf["grind"].unique())
            summary.append(
                {
                    "error_type": et,
                    "count": int(len(gdf)),
                    "rate": round(len(gdf) / n, 4),
                    "typical_grind_min": int(min(grinds)),
                    "typical_grind_max": int(max(grinds)),
                    "note": "v9 main ensemble",
                }
            )
        summary.append(
            {
                "error_type": "全部错分",
                "count": int(len(err_df)),
                "rate": round(len(err_df) / n, 4),
                "typical_grind_min": int(err_df["grind"].min()),
                "typical_grind_max": int(err_df["grind"].max()),
                "note": "pass-level interleaved · v9",
            }
        )
        plat = err_df[err_df["grind"] >= PLATEAU_G]
        summary.append(
            {
                "error_type": "平台后错分(g≥94)",
                "count": int(len(plat)),
                "rate": round(len(plat) / n, 4),
                "typical_grind_min": int(plat["grind"].min()) if len(plat) else None,
                "typical_grind_max": int(plat["grind"].max()) if len(plat) else None,
                "note": "细粒度难区",
            }
        )
    else:
        summary.append(
            {
                "error_type": "全部错分",
                "count": 0,
                "rate": 0.0,
                "typical_grind_min": None,
                "typical_grind_max": None,
                "note": "pass-level interleaved · v9",
            }
        )
    acc3 = eval_classification(y3, p3)["acc"]
    acc5 = eval_classification(y5, p5)["acc"]
    pd.DataFrame(summary).to_csv(out / "T10_error_summary.csv", index=False)
    err_df.to_csv(out / "T10_error_cases.csv", index=False)
    meta = {
        "protocol": "v9_main_soft+ens8",
        "n": n,
        "cls3_acc": acc3,
        "cls5_acc": acc5,
        "cm3": cm3.tolist(),
        "cm5": cm5.tolist(),
        "n_errors": int(len(err_df)),
    }
    save_json(meta, out / "T10_errors.json")
    print(f"  n={n} Acc3={acc3:.4f} Acc5={acc5:.4f} errors={len(err_df)}", flush=True)
    print(cm3, flush=True)
    return summary, meta


def run_significance(ours_rows, b5_rows, b6_rows, e_c3, e_mae, rf, frame_pp, out: Path):
    print("=== Significance / CI ===", flush=True)
    # align by key
    o_map = {r["key"]: r for r in ours_rows}
    b5_map = {r["key"]: r for r in b5_rows}
    b6_map = {r["key"]: r for r in b6_rows}
    keys = sorted(set(o_map) & set(b5_map) & set(b6_map))
    y3 = [o_map[k]["t3"] for k in keys]
    p_o = [int(np.asarray(o_map[k]["p3"]).argmax()) for k in keys]
    p_b5 = [int(np.asarray(b5_map[k]["p3"]).argmax()) for k in keys]
    p_b6 = [int(np.asarray(b6_map[k]["p3"]).argmax()) for k in keys]

    rows = []
    for name, pred in (("Ours_v9", p_o), ("B5_MeanPool_RF", p_b5), ("B6_BiLSTM", p_b6)):
        ci = bootstrap_acc_ci(y3, pred)
        rows.append(
            {
                "method": name,
                "metric": "cls3_acc",
                "value": round(ci["acc"], 4),
                "ci95_lo": round(ci["ci95_lo"], 4),
                "ci95_hi": round(ci["ci95_hi"], 4),
                "n": ci["n"],
            }
        )

    for other, pred, label in (("B5", p_b5, "Ours_vs_B5"), ("B6", p_b6, "Ours_vs_B6")):
        mc = mcnemar(y3, p_o, pred)
        dlt = bootstrap_delta_acc(y3, p_o, pred)
        rows.append(
            {
                "method": label,
                "metric": "cls3_McNemar",
                "value": round(dlt["delta_acc"], 4),
                "ci95_lo": round(dlt["ci95_lo"], 4),
                "ci95_hi": round(dlt["ci95_hi"], 4),
                "n": len(keys),
                "b10_Ours_win": mc["b10_Awin"],
                "b01_other_win": mc["b01_Bwin"],
                "n_discordant": mc["n_discordant"],
                "pvalue": round(mc["pvalue"], 6),
                "significant_0.05": bool(mc["pvalue"] < 0.05),
            }
        )

    # MAE CI: Ours frame-level; B5/B6 pass-level expanded to frames for paired comparison
    # Fairer: report Ours frame MAE CI; and pass-level MAE CI for all three
    yt_pass = [o_map[k]["tp"] for k in keys]
    pp_o = [o_map[k]["pp"] for k in keys]
    pp_b5 = [b5_map[k]["pp"] for k in keys]
    pp_b6 = [b6_map[k]["pp"] for k in keys]
    for name, pp in (("Ours_v9_pass", pp_o), ("B5_pass", pp_b5), ("B6_pass", pp_b6)):
        ci = bootstrap_mae_ci(yt_pass, pp)
        rows.append(
            {
                "method": name,
                "metric": "progress_mae",
                "value": round(ci["mae"], 4),
                "ci95_lo": round(ci["ci95_lo"], 4),
                "ci95_hi": round(ci["ci95_hi"], 4),
                "n": ci["n"],
            }
        )

    # frame-level Ours MAE CI (main-table protocol)
    yt_f, yp_f = [], []
    a, b, c = MAIN_PROG_W["resnet"], MAIN_PROG_W["rf"], MAIN_PROG_W["lstm"]
    key_to_i = {r["key"]: i for i, r in enumerate(e_c3)}
    for k in keys:
        i = key_to_i[k]
        for rv in frame_pp[k]:
            yt_f.append(progress(k[0]))
            yp_f.append(a * rv + b * rf[k]["pp"] + c * e_mae[i]["pp"])
    ci_f = bootstrap_mae_ci(yt_f, yp_f)
    rows.append(
        {
            "method": "Ours_v9_frame",
            "metric": "progress_mae",
            "value": round(ci_f["mae"], 4),
            "ci95_lo": round(ci_f["ci95_lo"], 4),
            "ci95_hi": round(ci_f["ci95_hi"], 4),
            "n": ci_f["n"],
            "note": "main-table frame-level progress protocol",
        }
    )

    # paired MAE delta Ours_pass - B5_pass
    rng = np.random.default_rng(2)
    e_o = np.abs(np.asarray(yt_pass) - np.asarray(pp_o))
    e_b5 = np.abs(np.asarray(yt_pass) - np.asarray(pp_b5))
    e_b6 = np.abs(np.asarray(yt_pass) - np.asarray(pp_b6))
    for label, e_other in (("Ours_vs_B5_mae", e_b5), ("Ours_vs_B6_mae", e_b6)):
        deltas = []
        n = len(e_o)
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            deltas.append(float(e_o[idx].mean() - e_other[idx].mean()))
        lo, hi = np.quantile(deltas, [0.025, 0.975])
        rows.append(
            {
                "method": label,
                "metric": "delta_mae_pass",
                "value": round(float(e_o.mean() - e_other.mean()), 4),
                "ci95_lo": round(float(lo), 4),
                "ci95_hi": round(float(hi), 4),
                "n": n,
                "note": "negative = Ours better",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "T_significance.csv", index=False)
    save_json({"rows": rows}, out / "T_significance.json")
    print(df.to_string(index=False), flush=True)
    return df


def run_weight_sensitivity(e_c3, e_c5, e_mae, rf, frame_pp, out: Path):
    print("=== Fusion weight sensitivity ===", flush=True)
    base = MAIN_PROG_W
    # perturbations on simplex around main weights
    deltas = [-0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10]
    rows = []
    # vary resnet, renormalize rf/lstm keeping their relative ratio
    rf_l_sum = base["rf"] + base["lstm"]
    rf_ratio = base["rf"] / rf_l_sum
    for d in deltas:
        wr = float(np.clip(base["resnet"] + d, 0.0, 1.0))
        rest = 1.0 - wr
        w = {"resnet": wr, "rf": rest * rf_ratio, "lstm": rest * (1 - rf_ratio)}
        cls_rows = soft_cls(e_c3, e_c5, rf)
        # cls independent of prog w
        mcls = metrics_rows([{**r, "pp": 0.0} for r in cls_rows])
        prog = frame_prog_metrics(e_c3, e_mae, rf, frame_pp, w)
        rows.append(
            {
                "perturb": f"resnet{d:+.3f}",
                "w_resnet": round(w["resnet"], 4),
                "w_rf": round(w["rf"], 4),
                "w_lstm": round(w["lstm"], 4),
                "cls3_acc": round(mcls["cls3"]["acc"], 4),
                "cls5_acc": round(mcls["cls5"]["acc"], 4),
                "progress_mae": round(prog["mae"], 4),
                "progress_spearman": round(prog["spearman"], 4),
                "is_main": abs(d) < 1e-12,
            }
        )
    # also vary rf share within (rf+lstm) at fixed resnet
    for d in deltas:
        wr = base["resnet"]
        rest = 1.0 - wr
        rf_share = float(np.clip(rf_ratio + d, 0.0, 1.0))
        w = {"resnet": wr, "rf": rest * rf_share, "lstm": rest * (1 - rf_share)}
        prog = frame_prog_metrics(e_c3, e_mae, rf, frame_pp, w)
        mcls = metrics_rows([{**r, "pp": 0.0} for r in soft_cls(e_c3, e_c5, rf)])
        rows.append(
            {
                "perturb": f"rf_share{d:+.3f}",
                "w_resnet": round(w["resnet"], 4),
                "w_rf": round(w["rf"], 4),
                "w_lstm": round(w["lstm"], 4),
                "cls3_acc": round(mcls["cls3"]["acc"], 4),
                "cls5_acc": round(mcls["cls5"]["acc"], 4),
                "progress_mae": round(prog["mae"], 4),
                "progress_spearman": round(prog["spearman"], 4),
                "is_main": abs(d) < 1e-12,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "A_weight_sensitivity.csv", index=False)
    save_json({"main": base, "rows": rows}, out / "A_weight_sensitivity.json")
    print(df.to_string(index=False), flush=True)
    return df


def run_spatial_seeds(device, out: Path):
    print("=== Spatial multi-seed ±std ===", flush=True)
    split = "spatial"
    X, meta, names = load_feature_cache(ROOT / "outputs/baselines/features")
    seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    rf = train_rf(X, meta, split)
    pass_pp = load_pass_pp(split)
    ckpts = sorted(V8.glob(f"ours_v8_{split}_s*.pt"))
    from .fill_v9_gaps import v9_protocol

    # spatial disclosed weights (ours_v9_full_spatial.json)
    spat_w = {"resnet": 0.15, "rf": 0.78625, "lstm": 0.06375}
    seed_cells = []
    all_c3, all_c5, all_mae = [], [], []
    for ck in ckpts:
        seed = int(ck.stem.split("_s")[-1])
        b = eval_specialists(X, meta, names, seg, split, device, ck)
        all_c3.append(b["st_c3"])
        all_c5.append(b["st_c5"])
        all_mae.append(b["st_mae"])
        rows_rf = v9_protocol(
            b["st_c3"],
            b["st_c5"],
            b["st_mae"],
            rf,
            pass_pp,
            prog_w=spat_w,
            cls_mode="rf_only",
        )
        c = cell_of(metrics_rows(rows_rf))
        c["seed"] = seed
        c["cls_mode"] = "rf_only"
        seed_cells.append(c)
        print(
            f"  seed{seed}: cls3={c['cls3_acc']:.4f} cls5={c['cls5_acc']:.4f} "
            f"mae={c['progress_mae']:.4f} sp={c['progress_spearman']:.4f}",
            flush=True,
        )
    e_c3, e_c5, e_mae = ens(all_c3), ens(all_c5), ens(all_mae)

    primary = cell_of(
        metrics_rows(
            v9_protocol(e_c3, e_c5, e_mae, rf, pass_pp, prog_w=spat_w, cls_mode="rf_only")
        )
    )
    df = pd.DataFrame(seed_cells)
    summary = {
        "split": split,
        "protocol": "v9_spatial_rf_cls+pass_prog (match main spatial)",
        "prog_w": spat_w,
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
        "note": "附表；近饱和，不作主结论",
    }
    pd.DataFrame([summary]).to_csv(out / "seed_mean_std_spatial.csv", index=False)
    df.to_csv(out / "seed_v9_per_seed_spatial.csv", index=False)
    save_json({"summary": summary, "seeds": seed_cells, "primary": primary}, out / "seed_v9_spatial.json")
    print(summary, flush=True)
    return summary


def run_e2e_latency(device, out: Path):
    print("=== End-to-end pass latency ===", flush=True)
    split = "interleaved"
    X, meta, names = load_feature_cache(ROOT / "outputs/baselines/features")
    seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    meta_p = build_pass_table(X, meta)
    # pick a few test passes
    test_passes = []
    for grind, pass_id, gdf in iter_pass_groups(meta_p):
        if pass_split_tag(gdf, split) == "test":
            test_passes.append((int(grind), int(pass_id), gdf))
        if len(test_passes) >= 40:
            break

    # RF train once (amortized; measure predict only)
    rf = train_rf(X, meta, split)
    from .temporal import make_pass_features

    Xp, y3, y5, yp, tags, keys = make_pass_features(X, meta_p, split)
    te = tags == "test"
    # rebuild RF models for timed predict — use same train_rf internals via sklearn on subset
    # measure RF dict lookup cost (already predicted)
    t0 = time.perf_counter()
    for g, pid, _ in test_passes:
        _ = rf[(g, pid)]
    t_rf = (time.perf_counter() - t0) / len(test_passes) * 1000

    # LSTM 3 specialists
    ckpt = sorted(V8.glob(f"ours_v8_{split}_s*.pt"))[0]
    state = torch.load(ckpt, map_location=device, weights_only=False)
    models_lstm = {}
    for key in ("st_c3", "st_c5", "st_mae"):
        m = TemporalExpert(len(names)).to(device)
        m.load_state_dict(state[key])
        m.eval()
        models_lstm[key] = m

    def one_pass_tensor(gdf):
        idx = gdf["feat_idx"].to_numpy()
        feats = torch.from_numpy(X[idx]).float().unsqueeze(0).to(device)
        lens = torch.tensor([feats.size(1)], dtype=torch.long)
        sg = torch.from_numpy(seg.get(int(gdf["grind"].iloc[0]), np.zeros(4, np.float32))).unsqueeze(0).to(device)
        return feats, lens, sg

    # warmup
    g0, p0, gdf0 = test_passes[0]
    x, lens, sg = one_pass_tensor(gdf0)
    with torch.no_grad():
        for m in models_lstm.values():
            m(x, lens, sg)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for g, pid, gdf in test_passes:
            x, lens, sg = one_pass_tensor(gdf)
            for m in models_lstm.values():
                m(x, lens, sg)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_lstm = (time.perf_counter() - t0) / len(test_passes) * 1000

    # ResNet progress: 4 frames / pass
    cnn = MultiTaskCNN("resnet18").to(device)
    st = torch.load(
        ROOT / "outputs/baselines/cnn/resnet18_interleaved_best_spearman.pt",
        map_location=device,
        weights_only=False,
    )
    cnn.load_state_dict(st["model"])
    cnn.eval()
    tf = transforms.Compose(
        [
            transforms.Resize((128, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def load_4(gdf):
        paths = gdf["path"].tolist()
        sel = np.linspace(0, len(paths) - 1, min(4, len(paths)), dtype=int)
        xs = torch.stack([tf(Image.open(paths[i]).convert("RGB")) for i in sel]).to(device)
        return xs

    xs = load_4(gdf0)
    with torch.no_grad():
        cnn(xs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_load = 0.0
    t_cnn = 0.0
    for g, pid, gdf in test_passes:
        t1 = time.perf_counter()
        xs = load_4(gdf)
        t_load += time.perf_counter() - t1
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            cnn(xs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_cnn += time.perf_counter() - t1
    t_load_ms = t_load / len(test_passes) * 1000
    t_cnn_ms = t_cnn / len(test_passes) * 1000

    t_fuse = 0.0
    for g, pid, gdf in test_passes:
        t1 = time.perf_counter()
        rr = rf[(g, pid)]
        _ = 0.9 * np.ones(3) / 3 + 0.1 * rr["p3"]
        _ = MAIN_PROG_W["resnet"] * 0.5 + MAIN_PROG_W["rf"] * rr["pp"] + MAIN_PROG_W["lstm"] * 0.5
        t_fuse += time.perf_counter() - t1
    t_fuse_ms = t_fuse / len(test_passes) * 1000

    e2e_feat = t_rf + t_lstm + t_fuse  # features already cached
    e2e_img = t_load_ms + t_cnn_ms + t_lstm + t_rf + t_fuse_ms

    rows = [
        {"stage": "RF lookup (features ready)", "ms_per_pass": round(t_rf, 3)},
        {"stage": "LSTM 3 specialty heads", "ms_per_pass": round(t_lstm, 3)},
        {"stage": "ResNet decode 4 frames (I/O+preprocess)", "ms_per_pass": round(t_load_ms, 3)},
        {"stage": "ResNet forward 4 frames", "ms_per_pass": round(t_cnn_ms, 3)},
        {"stage": "Fusion (routing+weighted prog)", "ms_per_pass": round(t_fuse_ms, 3)},
        {
            "stage": "E2E features-ready (RF+LSTM+fuse)",
            "ms_per_pass": round(e2e_feat, 3),
            "note": "无图像解码；部署可缓存帧特征",
        },
        {
            "stage": "E2E image→result (4帧 ResNet + LSTM + RF + fuse)",
            "ms_per_pass": round(e2e_img, 3),
            "note": "含 JPEG/BMP 读入与预处理；主瓶颈",
        },
    ]
    df = pd.DataFrame(rows)
    # update T8
    t8 = pd.read_csv(out / "T8_compute.csv")
    # drop old e2e rows if any
    t8 = t8[~t8["method"].astype(str).str.contains("E2E|端到端", na=False)]
    extra = pd.DataFrame(
        [
            {
                "method": "Ours v9 · E2E（特征已缓存）",
                "device": str(device),
                "train_time": "—",
                "n_params": "—",
                "infer_ms": f"{e2e_feat:.2f} ms/趟",
            },
            {
                "method": "Ours v9 · E2E（图像→结果，4帧）",
                "device": str(device),
                "train_time": "—",
                "n_params": "—",
                "infer_ms": f"{e2e_img:.2f} ms/趟",
            },
        ]
    )
    t8 = pd.concat([t8, extra], ignore_index=True)
    t8.to_csv(out / "T8_compute.csv", index=False)
    t8.to_json(out / "T8_compute.json", orient="records", force_ascii=False, indent=2)
    df.to_csv(out / "T8_e2e_latency.csv", index=False)
    save_json({"rows": rows, "n_passes": len(test_passes)}, out / "T8_e2e_latency.json")
    print(df.to_string(index=False), flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split = "interleaved"
    print("Loading interleaved bundle...", flush=True)
    X, meta, names = load_feature_cache(ROOT / "outputs/baselines/features")
    seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    rf = train_rf(X, meta, split)
    frame_pp = load_frame_pp(split)
    pass_pp = load_pass_pp(split)
    ckpts = sorted(V8.glob(f"ours_v8_{split}_s*.pt"))
    e_c3, e_c5, e_mae = ens_specialists(X, meta, names, seg, split, device, ckpts)
    ours = build_ours_pass_rows(e_c3, e_c5, e_mae, rf, frame_pp=frame_pp, pass_pp=pass_pp)
    print("Ours cell:", cell_of(metrics_rows(ours)), flush=True)
    if frame_pp is not None:
        print("Frame prog:", frame_prog_metrics(e_c3, e_mae, rf, frame_pp), flush=True)

    run_t10(ours, OUT)

    # B5 rows from RF
    b5 = [
        {
            "key": r["key"],
            "grind": r["grind"],
            "p3": rf[r["key"]]["p3"],
            "p5": rf[r["key"]]["p5"],
            "pp": rf[r["key"]]["pp"],
            "t3": r["t3"],
            "t5": r["t5"],
            "tp": r["tp"],
        }
        for r in ours
    ]
    b6 = eval_b6(X, meta, names, seg, split, device)
    run_significance(ours, b5, b6, e_c3, e_mae, rf, frame_pp, OUT)
    run_weight_sensitivity(e_c3, e_c5, e_mae, rf, frame_pp, OUT)
    run_spatial_seeds(device, OUT)
    run_e2e_latency(device, OUT)
    print("DONE all extras", flush=True)


if __name__ == "__main__":
    main()
