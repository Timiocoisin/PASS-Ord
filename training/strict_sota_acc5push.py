#!/usr/bin/env python3
"""Strict nested SOTA push: Acc3=ens CNN; Acc5=val-selected stronger tabular expert.

Selection:
  - Train tabular Acc5 candidates on nested-train only; pick by val Acc5 (then MAE).
  - Require val Acc5 >= RF-baseline val Acc5.
  - Refit winner on train∪val; evaluate once on test.
  - Acc3 from 4-seed ResNet ens (cached packs); progress from CNN ens.
Fair baselines: B3=single ResNet refit; B5=RF(400,d16,seed=42).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from .data import load_frames_meta
from .features import load_feature_cache
from .metrics import eval_classification, eval_regression, save_json
from .nested_sota import (
    OUT,
    ROOT,
    cell_of,
    ens,
    eval_specialists,
    load_seg,
    metrics_from_rows,
)
from .strict_sota_stack import cnn_pass_packs, rf_map
from .temporal import annotate_nested_split, build_pass_table, make_pass_features

CACHE = OUT / "pack_cache"


def fit_predict_tabular(name, Xp, y3, y5, yp, fit, mask, seed=0):
    if name == "rf":
        c3 = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=seed)
        c5 = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=seed)
        rg = RandomForestRegressor(400, max_depth=16, n_jobs=-1, random_state=seed)
    elif name == "rf_deep":
        c3 = RandomForestClassifier(800, max_depth=24, n_jobs=-1, random_state=seed)
        c5 = RandomForestClassifier(800, max_depth=24, n_jobs=-1, random_state=seed)
        rg = RandomForestRegressor(800, max_depth=24, n_jobs=-1, random_state=seed)
    elif name == "rf_shallow":
        c3 = RandomForestClassifier(600, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        c5 = RandomForestClassifier(600, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        rg = RandomForestRegressor(600, max_depth=12, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    elif name == "et":
        c3 = ExtraTreesClassifier(600, max_depth=20, n_jobs=-1, random_state=seed)
        c5 = ExtraTreesClassifier(600, max_depth=20, n_jobs=-1, random_state=seed)
        rg = ExtraTreesRegressor(600, max_depth=20, n_jobs=-1, random_state=seed)
    elif name == "hgb":
        c3 = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=300, random_state=seed)
        c5 = HistGradientBoostingClassifier(max_depth=8, learning_rate=0.05, max_iter=300, random_state=seed)
        rg = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05, max_iter=300, random_state=seed)
    elif name == "hgb_fast":
        c3 = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=200, random_state=seed)
        c5 = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=200, random_state=seed)
        rg = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=200, random_state=seed)
    elif name == "gb":
        c3 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
        c5 = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
        rg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
    else:
        raise ValueError(name)

    c3.fit(Xp[fit], y3[fit])
    c5.fit(Xp[fit], y5[fit])
    rg.fit(Xp[fit], yp[fit])

    def full(clf, proba, n):
        o = np.zeros((len(proba), n))
        for j, c in enumerate(clf.classes_):
            o[:, int(c)] = proba[:, j]
        return o

    p3 = full(c3, c3.predict_proba(Xp[mask]), 3)
    p5 = full(c5, c5.predict_proba(Xp[mask]), 5)
    pp = rg.predict(Xp[mask]).astype(float)
    return p3, p5, pp


def packs_from_arrays(keys, idxs, p3, p5, pp):
    return {
        keys[idxs[i]]: {"p3": p3[i], "p5": p5[i], "pp": float(pp[i])}
        for i in range(len(idxs))
    }


def ens_seeds(name, Xp, y3, y5, yp, fit, mask, keys, idxs, seeds):
    p3 = p5 = pp = None
    for s in seeds:
        a, b, c = fit_predict_tabular(name, Xp, y3, y5, yp, fit, mask, seed=s)
        p3 = a if p3 is None else p3 + a
        p5 = b if p5 is None else p5 + b
        pp = c if pp is None else pp + c
    n = float(len(seeds))
    return packs_from_arrays(keys, idxs, p3 / n, p5 / n, pp / n)


def main():
    t0 = time.time()
    import torch

    device = torch.device("cuda:0")
    X, meta, names = load_feature_cache(ROOT / "outputs/baselines/features")
    meta = annotate_nested_split(meta)
    frames = annotate_nested_split(
        load_frames_meta(ROOT / "outputs/baselines/features/frame_features_meta.csv")
    )
    seg = load_seg(ROOT / "outputs/processed/segmentation/grind_seg_metrics.csv")
    sel = OUT / "select"
    rn_ens = [OUT / f"resnet_nested_s{s}.pt" for s in range(4)]
    rn_single = OUT / "resnet18_nested_refit.pt"

    meta_p = build_pass_table(X, meta)
    Xp, y3, y5, yp, tags, keys = make_pass_features(X, meta_p, "nested")
    fit_tr = tags == "train"
    fit_trv = np.isin(tags, ["train", "val"])
    mask_va = tags == "val"
    mask_te = tags == "test"
    idxs_va = np.where(mask_va)[0]
    idxs_te = np.where(mask_te)[0]

    print("=== CNN packs ===", flush=True)
    cnn_te = cnn_pass_packs(rn_ens, frames, "test", device, cache_name="rn4ens")
    cnn_va = cnn_pass_packs(rn_ens, frames, "val", device, cache_name="rn4ens")
    cnn_b3 = cnn_pass_packs([rn_single], frames, "test", device, cache_name="rn_single")

    print("=== LSTM ===", flush=True)
    et_c3, et_c5, et_mae = [], [], []
    for s in range(8):
        b = eval_specialists(X, meta, names, seg, "test", device, sel / f"ours_nested_s{s}.pt")
        et_c3.append(b["st_c3"])
        et_c5.append(b["st_c5"])
        et_mae.append(b["st_mae"])
    et_c3, et_c5, et_mae = ens(et_c3), ens(et_c5), ens(et_mae)

    # B3 / B5 baselines
    b3_rows = [
        {
            **r,
            "p3": cnn_b3[r["key"]]["p3"],
            "p5": cnn_b3[r["key"]]["p5"],
            "pp": cnn_b3[r["key"]]["pp"],
        }
        for r in et_c3
    ]
    b3 = cell_of(metrics_from_rows(b3_rows))

    print("=== Acc5 expert search on val ===", flush=True)
    cand_names = ["rf", "rf_deep", "rf_shallow", "et", "hgb", "hgb_fast", "gb"]
    floor_packs = ens_seeds("rf", Xp, y3, y5, yp, fit_tr, mask_va, keys, idxs_va, (42,))
    y5v = np.array([y5[i] for i in idxs_va])
    floor_acc = eval_classification(
        y5v, np.array([floor_packs[keys[i]]["p5"].argmax() for i in idxs_va])
    )["acc"]
    print(f"RF seed42 val Acc5 floor={floor_acc:.4f}", flush=True)

    best = None
    val_leaderboard = {}
    for name in cand_names:
        for nseed, seeds in [(1, (0,)), (1, (42,)), (4, (0, 1, 2, 3)), (8, tuple(range(8)))]:
            # skip expensive combos for gb
            if name == "gb" and nseed > 1:
                continue
            if name.startswith("hgb") and nseed > 4:
                continue
            tag = f"{name}_s{nseed}_{seeds[0]}"
            packs = ens_seeds(name, Xp, y3, y5, yp, fit_tr, mask_va, keys, idxs_va, seeds)
            pred = np.array([packs[keys[i]]["p5"].argmax() for i in idxs_va])
            acc = eval_classification(y5v, pred)["acc"]
            mae = eval_regression(
                yp[idxs_va], np.array([packs[keys[i]]["pp"] for i in idxs_va])
            )["mae"]
            val_leaderboard[tag] = {"acc5": acc, "mae": mae, "name": name, "seeds": list(seeds)}
            print(f"  {tag}: Acc5={acc:.4f} MAE={mae:.4f}", flush=True)
            if acc + 1e-12 < floor_acc:
                continue
            score = (acc, -mae)
            if best is None or score > best[0]:
                best = (score, tag, name, seeds)

    if best is None:
        print("No tabular beat RF floor; using rf seed42", flush=True)
        best = ((floor_acc, 0.0), "rf_s1_42", "rf", (42,))
    print("Selected Acc5 expert", best[1], best[0], flush=True)

    # Refit winner on train∪val → test
    tab_te = ens_seeds(
        best[2], Xp, y3, y5, yp, fit_trv, mask_te, keys, idxs_te, best[3]
    )
    b5_te = ens_seeds("rf", Xp, y3, y5, yp, fit_trv, mask_te, keys, idxs_te, (42,))

    b5_rows = [
        {**r, "p3": b5_te[r["key"]]["p3"], "p5": b5_te[r["key"]]["p5"], "pp": b5_te[r["key"]]["pp"]}
        for r in et_c3
    ]
    b5 = cell_of(metrics_from_rows(b5_rows))

    # Ours: Acc3=CNN ens, Acc5=selected tab, prog=CNN ens (optionally blend with tab if val says so)
    # Progress blend on val: CNN vs selected tab
    tab_va = ens_seeds(best[2], Xp, y3, y5, yp, fit_tr, mask_va, keys, idxs_va, best[3])
    best_w = (1.0, 0.0)  # (w_cnn, w_tab)
    best_mae = None
    for wc in np.linspace(0.5, 1.0, 11):
        wt = 1 - wc
        pred = np.array(
            [
                wc * cnn_va[keys[i]]["pp"] + wt * tab_va[keys[i]]["pp"]
                for i in idxs_va
                if keys[i] in cnn_va
            ]
        )
        # align keys present in cnn_va
        ypv = []
        pred = []
        for i in idxs_va:
            k = keys[i]
            if k not in cnn_va:
                continue
            ypv.append(yp[i])
            pred.append(wc * cnn_va[k]["pp"] + wt * tab_va[k]["pp"])
        mae = eval_regression(np.array(ypv), np.array(pred))["mae"]
        if best_mae is None or mae < best_mae:
            best_mae = mae
            best_w = (wc, wt)
    print(f"prog blend val mae={best_mae:.5f} w_cnn={best_w[0]:.2f}", flush=True)

    rows = []
    for i, r in enumerate(et_c3):
        k = r["key"]
        pp = best_w[0] * cnn_te[k]["pp"] + best_w[1] * tab_te[k]["pp"]
        rows.append(
            {
                **r,
                "p3": cnn_te[k]["p3"],
                "p5": tab_te[k]["p5"],
                "pp": float(pp),
            }
        )
    ours = cell_of(metrics_from_rows(rows))
    print("B3", b3, flush=True)
    print("B5", b5, flush=True)
    print("OURS", ours, flush=True)

    sota = {
        "cls3_vs_B3": ours["cls3_acc"] > b3["cls3_acc"] + 1e-12,
        "cls5_vs_B5": ours["cls5_acc"] > b5["cls5_acc"] + 1e-12,
        "mae_vs_B3": ours["progress_mae"] < b3["progress_mae"] - 1e-12,
        "mae_vs_B5": ours["progress_mae"] < b5["progress_mae"] - 1e-12,
        "sp_vs_B3": ours["progress_spearman"] > b3["progress_spearman"] + 1e-12,
        "sp_vs_B5": ours["progress_spearman"] > b5["progress_spearman"] + 1e-12,
        "all": False,
    }
    sota["all"] = all(
        [
            sota["cls3_vs_B3"],
            sota["cls5_vs_B5"],
            sota["mae_vs_B3"],
            sota["mae_vs_B5"],
            sota["sp_vs_B3"],
            sota["sp_vs_B5"],
        ]
    )
    print("SOTA", sota, flush=True)

    out = {
        "protocol": "nested_val_select_stronger_acc5_expert",
        "acc5_expert": best[1],
        "prog_w": {"cnn": best_w[0], "tab": best_w[1]},
        "val_leaderboard": val_leaderboard,
        "baselines": {"B3_single": b3, "B5_rf42": b5},
        "ours": ours,
        "sota": sota,
        "elapsed_s": time.time() - t0,
    }
    save_json(out, OUT / "strict_sota_acc5push.json")
    print(f"Wrote {OUT / 'strict_sota_acc5push.json'}", flush=True)


if __name__ == "__main__":
    main()
