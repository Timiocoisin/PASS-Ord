#!/usr/bin/env python3
"""Refresh paper CSVs under nested maxconf SOTA protocol, then figures can be rebuilt."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from .data import FrameDataset, load_frames_meta
from .features import load_feature_cache
from .labels import cls3, cls5
from .metrics import eval_classification, eval_regression, save_json
from .nested_sota import (
    OUT,
    ROOT,
    MultiTaskCNN,
    cell_of,
    ens,
    eval_specialists,
    load_seg,
    metrics_from_rows,
)
from .strict_sota_stack import cnn_pass_packs, rf_map
from .temporal import annotate_nested_split

TAB = ROOT / "outputs/baselines/paper_tables"
FIG = ROOT / "outputs/baselines/paper_figures"
TAB.mkdir(parents=True, exist_ok=True)


def onehot(pred, n):
    o = np.zeros((len(pred), n), float)
    o[np.arange(len(pred)), pred] = 1.0
    return o


def bootstrap_acc(y, pred, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pred = np.asarray(pred)
    n_s = len(y)
    accs = []
    for _ in range(n):
        idx = rng.integers(0, n_s, n_s)
        accs.append((pred[idx] == y[idx]).mean())
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(np.mean(accs)), float(lo), float(hi)


def mcnemar_p(y, a, b):
    # exact binomial on discordant pairs
    from scipy.stats import binomtest

    y, a, b = map(np.asarray, (y, a, b))
    n01 = int(((a != y) & (b == y)).sum())
    n10 = int(((a == y) & (b != y)).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n10, n01 + n10, 0.5).pvalue)


def main():
    t0 = time.time()
    device = torch.device("cuda:0")
    X, meta, names = load_feature_cache(ROOT / "outputs/baselines/features")
    meta = annotate_nested_split(meta)
    frames = annotate_nested_split(
        load_frames_meta(ROOT / "outputs/baselines/features/frame_features_meta.csv")
    )
    seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    sel = OUT / "select"
    rn = [OUT / f"resnet_nested_s{s}.pt" for s in range(4)]
    rn1 = OUT / "resnet18_nested_refit.pt"

    print("packs...", flush=True)
    cnn_te = cnn_pass_packs(rn, frames, "test", device, cache_name="rn4ens")
    # per-seed CNN packs for seed table
    cnn_seeds = []
    for s, ck in enumerate(rn):
        cnn_seeds.append(
            cnn_pass_packs([ck], frames, "test", device, cache_name=f"rn_s{s}")
        )
    cnn_b3 = cnn_pass_packs([rn1], frames, "test", device, cache_name="rn_single")
    rf_te = rf_map(X, meta, ("train", "val"), "test", seeds=(42,))

    print("LSTM ens...", flush=True)
    all_c3, all_c5, all_mae = [], [], []
    for s in range(8):
        b = eval_specialists(
            X, meta, names, seg, "test", device, sel / f"ours_nested_s{s}.pt"
        )
        all_c3.append(b["st_c3"])
        all_c5.append(b["st_c5"])
        all_mae.append(b["st_mae"])
    et_c3, et_c5, et_mae = ens(all_c3), ens(all_c5), ens(all_mae)

    keys = [r["key"] for r in et_c3]
    y3 = np.array([r["t3"] for r in et_c3])
    y5 = np.array([r["t5"] for r in et_c3])
    yp = np.array([r["tp"] for r in et_c3], float)

    def maxconf_p5(k):
        c, r = cnn_te[k]["p5"], rf_te[k]["p5"]
        return c if c.max() >= r.max() else r

    def rows_variant(mode: str):
        rows = []
        for i, r in enumerate(et_c3):
            k = r["key"]
            if mode == "full":
                p3, p5, pp = cnn_te[k]["p3"], maxconf_p5(k), cnn_te[k]["pp"]
            elif mode == "w/o_ResNet_progress":
                p3, p5 = cnn_te[k]["p3"], maxconf_p5(k)
                pp = 0.5 * rf_te[k]["pp"] + 0.5 * et_mae[i]["pp"]
            elif mode == "w/o_task_routing":
                p3 = (et_c3[i]["p3"] + rf_te[k]["p3"] + cnn_te[k]["p3"]) / 3.0
                p5 = (et_c5[i]["p5"] + rf_te[k]["p5"] + cnn_te[k]["p5"]) / 3.0
                pp = cnn_te[k]["pp"]
            elif mode == "single_expert_LSTM":
                p3, p5, pp = et_c3[i]["p3"], et_c5[i]["p5"], et_mae[i]["pp"]
            elif mode == "single_expert_RF":
                p3, p5, pp = rf_te[k]["p3"], rf_te[k]["p5"], rf_te[k]["pp"]
            elif mode == "progress_ResNet_only":
                p3, p5, pp = cnn_te[k]["p3"], maxconf_p5(k), cnn_te[k]["pp"]
            elif mode == "w/o_RF_in_cls":
                p3, p5, pp = cnn_te[k]["p3"], cnn_te[k]["p5"], cnn_te[k]["pp"]
            elif mode == "w/o_stitch_seg_prior":
                # current pipeline does not inject stitch at inference; identical to full
                p3, p5, pp = cnn_te[k]["p3"], maxconf_p5(k), cnn_te[k]["pp"]
            elif mode == "CNN_only_cls":
                p3, p5, pp = cnn_te[k]["p3"], cnn_te[k]["p5"], cnn_te[k]["pp"]
            else:
                raise ValueError(mode)
            rows.append({**r, "p3": p3, "p5": p5, "pp": float(pp)})
        return rows

    # --- Full / baselines ---
    full = cell_of(metrics_from_rows(rows_variant("full")))
    b3 = cell_of(
        metrics_from_rows(
            [
                {
                    **r,
                    "p3": cnn_b3[r["key"]]["p3"],
                    "p5": cnn_b3[r["key"]]["p5"],
                    "pp": cnn_b3[r["key"]]["pp"],
                }
                for r in et_c3
            ]
        )
    )
    b5 = cell_of(
        metrics_from_rows(
            [
                {
                    **r,
                    "p3": rf_te[r["key"]]["p3"],
                    "p5": rf_te[r["key"]]["p5"],
                    "pp": rf_te[r["key"]]["pp"],
                }
                for r in et_c3
            ]
        )
    )

    # Frame MAE for full / B3
    def frame_prog(ckpts):
        ds = FrameDataset(frames, "nested", "test", train=False)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        models = []
        for ck in ckpts:
            m = MultiTaskCNN().to(device)
            m.load_state_dict(
                torch.load(ck, map_location=device, weights_only=False)["model"]
            )
            m.eval()
            models.append(m)
        preds, ys, gs = [], [], []
        with torch.no_grad():
            for bi, (x, y3b, y5b, yp_b) in enumerate(loader):
                x = x.to(device)
                outs = [m(x)[2].squeeze(-1).cpu().numpy() for m in models]
                preds.append(np.mean(outs, 0))
                ys.append(yp_b.numpy())
        yt = np.concatenate(ys)
        yp_h = np.concatenate(preds)
        return yt, yp_h, eval_regression(yt, yp_h)

    print("frame progress...", flush=True)
    yt_f, yp_f, fr_ours = frame_prog(rn)
    yt_b3, yp_b3, fr_b3 = frame_prog([rn1])
    # cache for Fig06
    # Fig06 expects keys yt/yp under paper_figures/_progress_points.npz
    np.savez(FIG / "_progress_points.npz", yt=yt_f, yp=yp_f)

    # --- Ablations ---
    abl_modes = [
        ("Ours_v9_full_main", "full", "nested maxconf Acc5 + RN4 Acc3/prog"),
        ("w/o_ResNet_progress", "w/o_ResNet_progress", "progress = RF∥LSTM only"),
        ("w/o_task_routing", "w/o_task_routing", "equal soft fuse L/R/C"),
        ("single_expert_LSTM", "single_expert_LSTM", "LSTM only"),
        ("single_expert_RF", "single_expert_RF", "RF only ≈ B5"),
        ("progress_ResNet_only", "progress_ResNet_only", "progress = ResNet ens"),
        ("w/o_RF_in_cls", "w/o_RF_in_cls", "cls without RF (CNN Acc5)"),
        ("w/o_stitch_seg_prior", "w/o_stitch_seg_prior", "identical (no stitch at infer)"),
    ]
    abl_rows = []
    abl_cells = {}
    for name, mode, note in abl_modes:
        cell = cell_of(metrics_from_rows(rows_variant(mode)))
        # use frame MAE for full / progress_ResNet_only / stitch; pass MAE otherwise for progress-ablated
        if mode in ("full", "progress_ResNet_only", "w/o_stitch_seg_prior", "w/o_task_routing", "w/o_RF_in_cls"):
            mae, sp = fr_ours["mae"], fr_ours["spearman"]
            if mode == "w/o_ResNet_progress":
                mae, sp = cell["progress_mae"], cell["progress_spearman"]
            elif mode not in ("full", "progress_ResNet_only", "w/o_stitch_seg_prior"):
                # classification-changed variants keep frame CNN progress
                mae, sp = fr_ours["mae"], fr_ours["spearman"]
        else:
            mae, sp = cell["progress_mae"], cell["progress_spearman"]
        if mode == "w/o_ResNet_progress":
            mae, sp = cell["progress_mae"], cell["progress_spearman"]
        abl_cells[name] = cell
        abl_rows.append(
            {
                "variant": name,
                "cls3_acc": cell["cls3_acc"],
                "cls3_f1": cell["cls3_f1"],
                "cls5_acc": cell["cls5_acc"],
                "cls5_f1": cell["cls5_f1"],
                "progress_mae": mae,
                "progress_spearman": sp,
                "plateau_acc": cell["plateau_acc"],
                "note": note,
            }
        )
    # fix full / progress_ResNet / stitch to frame metrics
    for nm in ("Ours_v9_full_main", "progress_ResNet_only", "w/o_stitch_seg_prior"):
        for row in abl_rows:
            if row["variant"] == nm:
                row["progress_mae"] = fr_ours["mae"]
                row["progress_spearman"] = fr_ours["spearman"]
    # w/o_task_routing and w/o_RF keep frame CNN progress
    for nm in ("w/o_task_routing", "w/o_RF_in_cls"):
        for row in abl_rows:
            if row["variant"] == nm:
                row["progress_mae"] = fr_ours["mae"]
                row["progress_spearman"] = fr_ours["spearman"]

    import pandas as pd

    pd.DataFrame(abl_rows).to_csv(TAB / "T6_ablation.csv", index=False)
    save_json({"rows": abl_rows, "full_pass": full, "frame": fr_ours}, TAB / "T6_ablation.json")

    # --- Confusion ---
    pred3 = np.array([cnn_te[k]["p3"].argmax() for k in keys])
    pred5 = np.array([maxconf_p5(k).argmax() for k in keys])
    cm3 = confusion_matrix(y3, pred3, labels=[0, 1, 2])
    cm5 = confusion_matrix(y5, pred5, labels=[0, 1, 2, 3, 4])
    pd.DataFrame(cm3, index=["Early", "Mid", "Late"], columns=["Early", "Mid", "Late"]).to_csv(
        TAB / "T10_cm_cls3.csv"
    )
    pd.DataFrame(
        cm5,
        index=[f"Bin{i}" for i in range(1, 6)],
        columns=[f"Bin{i}" for i in range(1, 6)],
    ).to_csv(TAB / "T10_cm_cls5.csv")

    # error structure text helpers
    err_idx = np.where(pred3 != y3)[0]
    err_g = [keys[i][0] for i in err_idx]
    err_summary = {
        "n_test": int(len(y3)),
        "n_err_cls3": int(len(err_idx)),
        "err_rate": float(len(err_idx) / len(y3)),
        "cm3": cm3.tolist(),
        "cm5": cm5.tolist(),
        "err_grinds": err_g,
        "acc3": float((pred3 == y3).mean()),
        "acc5": float((pred5 == y5).mean()),
    }
    save_json(err_summary, TAB / "T10_errors.json")
    # per-error cases
    pd.DataFrame(
        [
            {
                "grind": keys[i][0],
                "pass_id": keys[i][1],
                "y3": int(y3[i]),
                "pred3": int(pred3[i]),
                "y5": int(y5[i]),
                "pred5": int(pred5[i]),
            }
            for i in err_idx
        ]
    ).to_csv(TAB / "T10_error_cases.csv", index=False)

    # --- Per-seed ResNet (4 seeds): Acc3/Acc5/prog vs RF maxconf with that seed ---
    seed_rows = []
    for s, pack in enumerate(cnn_seeds):
        p3 = np.array([pack[k]["p3"].argmax() for k in keys])
        # Acc5: maxconf of this seed CNN vs RF
        p5 = []
        pps = []
        for k in keys:
            c, r = pack[k]["p5"], rf_te[k]["p5"]
            p5.append((c if c.max() >= r.max() else r).argmax())
            pps.append(pack[k]["pp"])
        p5 = np.array(p5)
        pps = np.array(pps)
        m3 = eval_classification(y3, p3)
        m5 = eval_classification(y5, p5)
        mp = eval_regression(yp, pps)
        seed_rows.append(
            {
                "seed": s,
                "cls3_acc": m3["acc"],
                "cls3_f1": m3["macro_f1"],
                "cls5_acc": m5["acc"],
                "cls5_f1": m5["macro_f1"],
                "progress_mae": mp["mae"],
                "progress_spearman": mp["spearman"],
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(TAB / "seed_v9_per_seed.csv", index=False)
    seed_df.to_csv(TAB / "nested_per_seed.csv", index=False)
    mean_row = {
        "cls3_acc_mean": float(seed_df.cls3_acc.mean()),
        "cls3_acc_std": float(seed_df.cls3_acc.std(ddof=1)),
        "cls5_acc_mean": float(seed_df.cls5_acc.mean()),
        "cls5_acc_std": float(seed_df.cls5_acc.std(ddof=1)),
        "progress_mae_mean": float(seed_df.progress_mae.mean()),
        "progress_mae_std": float(seed_df.progress_mae.std(ddof=1)),
        "progress_spearman_mean": float(seed_df.progress_spearman.mean()),
        "progress_spearman_std": float(seed_df.progress_spearman.std(ddof=1)),
        "n": len(seed_df),
        "ensemble_cls3": full["cls3_acc"],
        "ensemble_cls5": full["cls5_acc"],
        "ensemble_mae_frame": fr_ours["mae"],
        "ensemble_spearman_frame": fr_ours["spearman"],
    }
    pd.DataFrame([mean_row]).to_csv(TAB / "seed_mean_std.csv", index=False)
    save_json(mean_row, TAB / "seed_v9.json")

    # --- Significance: Acc3 vs B3, Acc5 vs B5 ---
    b3_p3 = np.array([cnn_b3[k]["p3"].argmax() for k in keys])
    b5_p5 = np.array([rf_te[k]["p5"].argmax() for k in keys])
    ours_p3, ours_p5 = pred3, pred5
    sig = []
    for name, y, ours_p, base_p in [
        ("Acc3_vs_B3", y3, ours_p3, b3_p3),
        ("Acc5_vs_B5", y5, ours_p5, b5_p5),
    ]:
        acc_o, lo_o, hi_o = bootstrap_acc(y, ours_p)
        acc_b, lo_b, hi_b = bootstrap_acc(y, base_p)
        # delta bootstrap
        rng = np.random.default_rng(0)
        deltas = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            deltas.append((ours_p[idx] == y[idx]).mean() - (base_p[idx] == y[idx]).mean())
        dlo, dhi = np.percentile(deltas, [2.5, 97.5])
        sig.append(
            {
                "contrast": name,
                "ours_acc": float((ours_p == y).mean()),
                "base_acc": float((base_p == y).mean()),
                "delta": float((ours_p == y).mean() - (base_p == y).mean()),
                "ours_ci_lo": lo_o,
                "ours_ci_hi": hi_o,
                "base_ci_lo": lo_b,
                "base_ci_hi": hi_b,
                "delta_ci_lo": float(dlo),
                "delta_ci_hi": float(dhi),
                "mcnemar_p": mcnemar_p(y, ours_p, base_p),
            }
        )
    pd.DataFrame(sig).to_csv(TAB / "T_significance.csv", index=False)
    save_json({"contrasts": sig}, TAB / "T_significance.json")

    # Weight sensitivity: Acc5 has no threshold; progress is pure CNN — report flat Acc3/Acc5
    # and optional soft Acc5 blends as diagnostic (not used in main)
    wrows = []
    for aR in [0.0, 0.25, 0.5, 0.75, 1.0]:
        aC = 1.0 - aR
        pred = []
        for k in keys:
            p = aR * rf_te[k]["p5"] + aC * cnn_te[k]["p5"]
            pred.append(p.argmax())
        acc = eval_classification(y5, np.array(pred))["acc"]
        wrows.append({"a5_rf": aR, "a5_cnn": aC, "cls5_acc": acc, "note": "soft Acc5 diagnostic"})
    # maxconf row
    wrows.append(
        {
            "a5_rf": -1,
            "a5_cnn": -1,
            "cls5_acc": full["cls5_acc"],
            "note": "maxconf (main)",
        }
    )
    pd.DataFrame(wrows).to_csv(TAB / "A_weight_sensitivity.csv", index=False)

    # Update T4 nested + interleaved ours/B3/B5
    t4n = pd.DataFrame(
        [
            {
                "model": "B3_ResNet18",
                "cls3_acc": b3["cls3_acc"],
                "cls3_f1": b3["cls3_f1"],
                "cls5_acc": b3["cls5_acc"],
                "cls5_f1": b3["cls5_f1"],
                "progress_mae": fr_b3["mae"],
                "progress_spearman": fr_b3["spearman"],
                "progress_level": "frame",
            },
            {
                "model": "B5_MeanPool_RF",
                "cls3_acc": b5["cls3_acc"],
                "cls3_f1": b5["cls3_f1"],
                "cls5_acc": b5["cls5_acc"],
                "cls5_f1": b5["cls5_f1"],
                "progress_mae": b5["progress_mae"],
                "progress_spearman": b5["progress_spearman"],
                "progress_level": "pass",
            },
            {
                "model": "Ours_PASSOrd",
                "cls3_acc": full["cls3_acc"],
                "cls3_f1": full["cls3_f1"],
                "cls5_acc": full["cls5_acc"],
                "cls5_f1": full["cls5_f1"],
                "progress_mae": fr_ours["mae"],
                "progress_spearman": fr_ours["spearman"],
                "progress_level": "frame",
            },
        ]
    )
    # keep B6 from prior nested file if present
    old = pd.read_csv(TAB / "T4_nested_main.csv")
    b6 = old[old.model == "B6_LSTM"]
    t4n = pd.concat([t4n.iloc[:2], b6, t4n.iloc[2:]], ignore_index=True)
    t4n.to_csv(TAB / "T4_nested_main.csv", index=False)

    summary = {
        "full_pass": full,
        "frame_ours": fr_ours,
        "frame_b3": fr_b3,
        "b3_pass": b3,
        "b5_pass": b5,
        "seed_mean": mean_row,
        "sig": sig,
        "err": err_summary,
        "elapsed_s": time.time() - t0,
    }
    save_json(summary, OUT / "paper_nested_numbers.json")
    print(json.dumps({k: summary[k] for k in ("full_pass", "frame_ours", "seed_mean")}, indent=2))
    print("err cls3", err_summary["n_err_cls3"], "grinds", sorted(set(err_g)))
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
