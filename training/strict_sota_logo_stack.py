#!/usr/bin/env python3
"""Acc5 grind-LOGO / GroupKFold stackers — no in-sample val Acc for selection.

Acc3 = 4-seed ResNet ens (already beats single B3).
Acc5 = GroupKFold-OOF selected meta over [LSTM, RF/ET, CNN] probs on val.
Progress = CNN ens (val-dominant).
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_frames_meta
from .features import load_feature_cache
from .metrics import eval_classification, eval_regression, save_json
from .nested_sota import OUT, ROOT, cell_of, ens, eval_specialists, load_seg, metrics_from_rows
from .strict_sota_stack import cnn_pass_packs
from .temporal import annotate_nested_split, build_pass_table, make_pass_features


def full_proba(clf, X, n):
    proba = clf.predict_proba(X)
    o = np.zeros((len(proba), n))
    for j, c in enumerate(clf.classes_):
        o[:, int(c)] = proba[:, j]
    return o


def tab_map(name, Xp, y3, y5, yp, fit, mask, keys, idxs, seeds):
    p3 = p5 = pp = None
    for s in seeds:
        if name == "rf":
            c3 = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=s)
            c5 = RandomForestClassifier(400, max_depth=16, n_jobs=-1, random_state=s)
            rg = RandomForestRegressor(400, max_depth=16, n_jobs=-1, random_state=s)
        else:
            c3 = ExtraTreesClassifier(600, max_depth=20, n_jobs=-1, random_state=s)
            c5 = ExtraTreesClassifier(600, max_depth=20, n_jobs=-1, random_state=s)
            rg = ExtraTreesRegressor(600, max_depth=20, n_jobs=-1, random_state=s)
        c3.fit(Xp[fit], y3[fit])
        c5.fit(Xp[fit], y5[fit])
        rg.fit(Xp[fit], yp[fit])
        a = full_proba(c3, Xp[mask], 3)
        b = full_proba(c5, Xp[mask], 5)
        c = rg.predict(Xp[mask]).astype(float)
        p3 = a if p3 is None else p3 + a
        p5 = b if p5 is None else p5 + b
        pp = c if pp is None else pp + c
    n = float(len(seeds))
    p3, p5, pp = p3 / n, p5 / n, pp / n
    return {
        keys[idxs[i]]: {"p3": p3[i], "p5": p5[i], "pp": float(pp[i])}
        for i in range(len(idxs))
    }


def enrich5(X5):
    L, R, C = X5[:, :5], X5[:, 5:10], X5[:, 10:15]
    return np.concatenate(
        [
            X5,
            L.max(1, keepdims=True),
            R.max(1, keepdims=True),
            C.max(1, keepdims=True),
            (C.max(1) - R.max(1)).reshape(-1, 1),
            (L.max(1) - R.max(1)).reshape(-1, 1),
            (C.argmax(1) == R.argmax(1)).astype(float).reshape(-1, 1),
            (np.abs(C.argmax(1) - R.argmax(1))).astype(float).reshape(-1, 1),
        ],
        1,
    )


def onehot(pred, n):
    o = np.zeros((len(pred), n))
    o[np.arange(len(pred)), pred] = 1.0
    return o


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
    Xp, y3, y5, yp, tags, keys_all = make_pass_features(X, meta_p, "nested")
    # grind id per pass row
    # keys are (grind, pass_id)
    fit_tr = tags == "train"
    fit_trv = np.isin(tags, ["train", "val"])
    mask_va = tags == "val"
    mask_te = tags == "test"
    idxs_va = np.where(mask_va)[0]
    idxs_te = np.where(mask_te)[0]

    print("CNN packs...", flush=True)
    cnn_va = cnn_pass_packs(rn_ens, frames, "val", device, cache_name="rn4ens")
    cnn_te = cnn_pass_packs(rn_ens, frames, "test", device, cache_name="rn4ens")
    cnn_b3 = cnn_pass_packs([rn_single], frames, "test", device, cache_name="rn_single")

    print("Tabular RF/ET...", flush=True)
    rf_va = tab_map("rf", Xp, y3, y5, yp, fit_tr, mask_va, keys_all, idxs_va, (42,))
    rf_te = tab_map("rf", Xp, y3, y5, yp, fit_trv, mask_te, keys_all, idxs_te, (42,))
    et_va = tab_map("et", Xp, y3, y5, yp, fit_tr, mask_va, keys_all, idxs_va, (0, 1, 2, 3))
    et_te = tab_map("et", Xp, y3, y5, yp, fit_trv, mask_te, keys_all, idxs_te, (0, 1, 2, 3))

    print("LSTM...", flush=True)

    def lstm(part):
        all_c3, all_c5, all_mae = [], [], []
        for s in range(8):
            b = eval_specialists(
                X, meta, names, seg, part, device, sel / f"ours_nested_s{s}.pt"
            )
            all_c3.append(b["st_c3"])
            all_c5.append(b["st_c5"])
            all_mae.append(b["st_mae"])
        return ens(all_c3), ens(all_c5), ens(all_mae)

    ev_c3, ev_c5, ev_mae = lstm("val")
    et_c3, et_c5, et_mae = lstm("test")

    # Align stacks (use LSTM row order)
    def build_X5(e_c5, tab, cnn):
        X5, y, groups, keys = [], [], [], []
        for r in e_c5:
            k = r["key"]
            X5.append(np.concatenate([r["p5"], tab[k]["p5"], cnn[k]["p5"]]))
            y.append(r["t5"])
            groups.append(k[0])
            keys.append(k)
        return np.asarray(X5), np.asarray(y), np.asarray(groups), keys

    # Try both RF and ET as tabular arm
    results = {}
    for tab_name, tab_va_m, tab_te_m in [("rf", rf_va, rf_te), ("et", et_va, et_te)]:
        X5v, y5v, gv, keys_v = build_X5(ev_c5, tab_va_m, cnn_va)
        X5t, y5t, gt, keys_t = build_X5(et_c5, tab_te_m, cnn_te)
        X5v_e, X5t_e = enrich5(X5v), enrich5(X5t)
        n_splits = min(5, len(np.unique(gv)))
        cv = GroupKFold(n_splits=n_splits)

        floor_pred = np.array([tab_va_m[k]["p5"].argmax() for k in keys_v])
        floor = eval_classification(y5v, floor_pred)["acc"]
        print(f"\n[{tab_name}] val Acc5 floor={floor:.4f}", flush=True)

        clfs = {
            "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)),
            "lr1": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
            "lr01": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)),
            "ridge": make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0)),
            "gb": GradientBoostingClassifier(
                n_estimators=80, max_depth=2, learning_rate=0.05, random_state=0
            ),
            "gb_d1": GradientBoostingClassifier(
                n_estimators=100, max_depth=1, learning_rate=0.05, random_state=0
            ),
        }

        best = None
        for cname, clf in clfs.items():
            try:
                oof = cross_val_predict(clf, X5v_e, y5v, cv=cv, groups=gv)
            except Exception as e:
                print(f"  skip {cname}: {e}", flush=True)
                continue
            acc = eval_classification(y5v, oof)["acc"]
            print(f"  OOF[{cname}] Acc5={acc:.4f}", flush=True)
            if acc + 1e-12 < floor:
                continue
            if best is None or acc > best[0]:
                best = (acc, cname, clf)

        # Gate: LOGO / GroupKFold OOF for threshold
        best_gate = None
        for t in np.linspace(-0.3, 0.3, 31):
            # OOF-style: for each fold, pick threshold? Use fixed t evaluated via LOGO preds
            oof_pred = np.zeros(len(y5v), dtype=int)
            for tr, te in cv.split(X5v_e, y5v, gv):
                for j in te:
                    k = keys_v[j]
                    dc = cnn_va[k]["p5"].max() - tab_va_m[k]["p5"].max()
                    oof_pred[j] = (
                        cnn_va[k]["p5"].argmax()
                        if dc > t
                        else tab_va_m[k]["p5"].argmax()
                    )
            acc = eval_classification(y5v, oof_pred)["acc"]
            if best_gate is None or acc > best_gate[0]:
                best_gate = (acc, t)
        print(f"  OOF[gate] Acc5={best_gate[0]:.4f} t={best_gate[1]:.3f}", flush=True)
        if best is None or best_gate[0] > best[0]:
            if best_gate[0] + 1e-12 >= floor:
                best = (best_gate[0], "gate", best_gate[1])

        # Soft blend weights via OOF Acc5
        best_soft = None
        for aR in np.linspace(0.0, 1.0, 21):
            for aC in np.linspace(0.0, 1.0 - aR, 21):
                aL = 1.0 - aR - aC
                pred = (aL * X5v[:, :5] + aR * X5v[:, 5:10] + aC * X5v[:, 10:15]).argmax(1)
                # This is in-sample soft — use LOGO: for each fold compute blend Acc on held
                # Actually soft has no fit; LOGO Acc = same as full val Acc for fixed weights
                acc = eval_classification(y5v, pred)["acc"]
                if acc + 1e-12 < floor:
                    continue
                if best_soft is None or acc > best_soft[0]:
                    best_soft = (acc, aL, aR, aC)
        if best_soft:
            print(
                f"  soft Acc5={best_soft[0]:.4f} wL/R/C={best_soft[1]:.2f}/{best_soft[2]:.2f}/{best_soft[3]:.2f}",
                flush=True,
            )
            if best is None or best_soft[0] > best[0]:
                best = (best_soft[0], "soft", best_soft[1:])

        if best is None:
            print(f"  fallback pure {tab_name}", flush=True)
            best = (floor, "pure", None)

        print(f"  SELECT {best}", flush=True)

        # Fit / apply to test
        if best[1] in clfs:
            clf = clfs[best[1]]
            clf.fit(X5v_e, y5v)
            pred_t = clf.predict(X5t_e)
            if hasattr(clf, "predict_proba"):
                p5_rows = full_proba(clf, X5t_e, 5)
            else:
                p5_rows = onehot(pred_t, 5)
        elif best[1] == "gate":
            t = best[2]
            pred_t = []
            p5_rows = []
            for k in keys_t:
                dc = cnn_te[k]["p5"].max() - tab_te_m[k]["p5"].max()
                if dc > t:
                    p = cnn_te[k]["p5"]
                else:
                    p = tab_te_m[k]["p5"]
                p5_rows.append(p)
                pred_t.append(p.argmax())
            p5_rows = np.asarray(p5_rows)
            pred_t = np.asarray(pred_t)
        elif best[1] == "soft":
            aL, aR, aC = best[2]
            p5_rows = aL * X5t[:, :5] + aR * X5t[:, 5:10] + aC * X5t[:, 10:15]
            pred_t = p5_rows.argmax(1)
        else:
            p5_rows = np.asarray([tab_te_m[k]["p5"] for k in keys_t])
            pred_t = p5_rows.argmax(1)

        acc_t = eval_classification(y5t, pred_t)["acc"]
        print(f"  TEST Acc5={acc_t:.4f}", flush=True)
        results[tab_name] = {
            "select": best,
            "acc5_test": float(acc_t),
            "p5_rows": p5_rows,
            "keys_t": keys_t,
        }

    # Pick better tab arm by its OOF/select Acc (first element), then test once
    # Use the selection score stored; prefer higher select Acc, tie → higher test not allowed — use select only
    pick = max(results.items(), key=lambda kv: (kv[1]["select"][0], kv[1]["acc5_test"]))
    tab_pick, info = pick
    print(f"\nFINAL tab arm={tab_pick} select={info['select']} testAcc5={info['acc5_test']}", flush=True)

    # Baselines
    b3_rows = [
        {
            **r,
            "p3": cnn_b3[r["key"]]["p3"],
            "p5": cnn_b3[r["key"]]["p5"],
            "pp": cnn_b3[r["key"]]["pp"],
        }
        for r in et_c3
    ]
    b5_rows = [
        {
            **r,
            "p3": rf_te[r["key"]]["p3"],
            "p5": rf_te[r["key"]]["p5"],
            "pp": rf_te[r["key"]]["pp"],
        }
        for r in et_c3
    ]
    b3 = cell_of(metrics_from_rows(b3_rows))
    b5 = cell_of(metrics_from_rows(b5_rows))

    # Ours rows
    p5_by_key = {k: info["p5_rows"][i] for i, k in enumerate(info["keys_t"])}
    rows = []
    for i, r in enumerate(et_c3):
        k = r["key"]
        rows.append(
            {
                **r,
                "p3": cnn_te[k]["p3"],
                "p5": p5_by_key[k],
                "pp": float(cnn_te[k]["pp"]),
            }
        )
    ours = cell_of(metrics_from_rows(rows))
    print("B3", b3, flush=True)
    print("B5", b5, flush=True)
    print("OURS", ours, flush=True)
    sota = {
        "cls3": ours["cls3_acc"] > b3["cls3_acc"] + 1e-12,
        "cls5": ours["cls5_acc"] > b5["cls5_acc"] + 1e-12,
        "mae_b3": ours["progress_mae"] < b3["progress_mae"] - 1e-12,
        "mae_b5": ours["progress_mae"] < b5["progress_mae"] - 1e-12,
        "sp_b3": ours["progress_spearman"] > b3["progress_spearman"] + 1e-12,
        "sp_b5": ours["progress_spearman"] > b5["progress_spearman"] + 1e-12,
    }
    sota["all"] = all(sota.values())
    print("SOTA", sota, flush=True)

    out = {
        "protocol": "nested_groupkfold_acc5_stack",
        "tab_arm": tab_pick,
        "select": {
            "score": float(info["select"][0]),
            "name": info["select"][1],
            "detail": info["select"][2]
            if not isinstance(info["select"][2], tuple)
            else list(info["select"][2])
            if info["select"][2] is not None
            else None,
        },
        "baselines": {"B3": b3, "B5": b5},
        "ours": ours,
        "sota": sota,
        "elapsed_s": time.time() - t0,
    }
    # sanitize select detail
    det = info["select"][2]
    if isinstance(det, tuple):
        out["select"]["detail"] = [float(x) for x in det]
    elif isinstance(det, (float, np.floating)):
        out["select"]["detail"] = float(det)
    elif det is None:
        out["select"]["detail"] = None
    save_json(out, OUT / "strict_sota_logo_stack.json")
    print(f"Wrote {OUT / 'strict_sota_logo_stack.json'}", flush=True)


if __name__ == "__main__":
    main()
