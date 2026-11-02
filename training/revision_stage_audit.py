"""CPU-only stage-aware reanalysis of retained predictions and feature caches.

No CNN is retrained. Buffered-block RF results are diagnostics, not PASS-Ord
generalization results. Bootstrap intervals condition on this single trajectory.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/baselines/revision_audit'

def clustered_interval(values, groups, seed=20261030, repetitions=10000):
    """Paired micro-average: resample whole stages, preserving all their rows."""
    values, groups = np.asarray(values), np.asarray(groups)
    levels = np.unique(groups)
    sums = np.array([values[groups == g].sum() for g in levels])
    counts = np.array([(groups == g).sum() for g in levels])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(levels), size=(repetitions, len(levels)))
    draws = sums[indices].sum(1) / counts[indices].sum(1)
    lo, hi = np.quantile(draws, [.025, .975])
    # Sensitivity to serial correlation across neighboring odd test stages.
    starts = rng.integers(0, len(levels), size=(repetitions, (len(levels)+4)//5))
    blocks = (starts[:,:,None]+np.arange(5))%len(levels)
    indices = blocks.reshape(repetitions,-1)[:,:len(levels)]
    block_draws = sums[indices].sum(1)/counts[indices].sum(1)
    blo,bhi = np.quantile(block_draws,[.025,.975])
    return {'estimate': float(values.mean()), 'ci95_lo': float(lo),
            'ci95_hi': float(hi), 'n_stages': len(levels),
            'n_observations': len(values), 'bootstrap_repetitions': repetitions,
            'circular_block_length_test_stages':5,
            'block_ci95_lo':float(blo),'block_ci95_hi':float(bhi)}

def labels(g):
    return np.where(g <= 40, 0, np.where(g <= 110, 1, 2)), (g-1)//34, g/169.

def forest_predict(X, g, fit, test):
    y3, y5, yp = labels(g)
    predictions = []
    for estimator, target in [(RandomForestClassifier, y3),
                              (RandomForestClassifier, y5),
                              (RandomForestRegressor, yp)]:
        model = estimator(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
        model.fit(X[fit], target[fit])
        predictions.append(model.predict(X[test]))
    # RF confidence comparison needs the full five-class probability vector.
    c5 = RandomForestClassifier(n_estimators=400, max_depth=16, n_jobs=-1, random_state=42)
    c5.fit(X[fit], y5[fit])
    probs = np.zeros((sum(test),5))
    probs[:,c5.classes_.astype(int)] = c5.predict_proba(X[test])
    return (*predictions, probs)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT/'outputs/baselines/features/frame_features_meta.csv')
    Xf = np.load(ROOT/'outputs/baselines/features/frame_features.npz')['X']
    keys, xp = [], []
    for key, df in meta.groupby(['grind','pass_id'],sort=True):
        x = Xf[df.index.to_numpy()]
        keys.append(tuple(map(int,key)))
        xp.append(np.concatenate([x.mean(0),x.std(0),x.max(0),x.min(0)]))
    X = np.asarray(xp,dtype=np.float32)
    g = np.array([k[0] for k in keys])
    fit, test = g%2==0, g%2==1
    rf3, rf5, rfp, rfp5 = forest_predict(X,g,fit,test)
    testkeys = [k for k,t in zip(keys,test) if t]
    cache = ROOT/'outputs/baselines/ours_nested/pack_cache'
    cnn = np.load(cache/'rn4ens_test.npy',allow_pickle=True).item()
    b3 = np.load(cache/'rn_single_test.npy',allow_pickle=True).item()
    cp3 = np.array([cnn[k]['p3'] for k in testkeys])
    cp5 = np.array([cnn[k]['p5'] for k in testkeys])
    op3 = cp3.argmax(1)
    op5 = np.where(cp5.max(1)>=rfp5.max(1),cp5.argmax(1),rf5)
    bp3 = np.array([b3[k]['p3'].argmax() for k in testkeys])
    yp3, yp5, target = labels(g[test])
    op = np.array([cnn[k]['pp'] for k in testkeys])
    bp = np.array([b3[k]['pp'] for k in testkeys])
    assert np.isclose((op3==yp3).mean(),.9909297052154195)
    assert np.isclose((op5==yp5).mean(),.981859410430839)
    metrics = []
    for name, values in [('Acc3_vs_B3',(op3==yp3).astype(float)-(bp3==yp3)),
                         ('Acc5_vs_B5',(op5==yp5).astype(float)-(rf5==yp5)),
                         ('passMAE_vs_B3',abs(bp-target)-abs(op-target))]:
        metrics.append({'contrast':name, **clustered_interval(values,g[test])})
    progress = np.load(ROOT/'outputs/baselines/paper_figures/_progress_points.npz')
    fg = np.rint(progress['yt']*169).astype(int)
    assert len(fg)==8736 and set(fg)==set(g[test])
    frame = clustered_interval(abs(progress['yp']-progress['yt']),fg)
    # Fixed diagnostic blocks; no parameter search on any held-out block.
    # Ten-stage interior windows retain every label class in the training set.
    # Holding out a whole quintile would make RF Acc5 trivially zero because
    # that class would be absent from fitting, rather than test visual transfer.
    bounds = [(16,25),(46,55),(81,90),(116,125),(151,160)]
    folds, predictions = [], []
    for fold,(lo,hi) in enumerate(bounds,1):
        te = (g>=lo)&(g<=hi)
        tr = (g<lo-2)|(g>hi+2)
        a,b,p,_ = forest_predict(X,g,tr,te)
        y3,y5,yp = labels(g[te])
        folds.append({'fold':fold,'test_start':lo,'test_end':hi,'buffer_stages':2,
                      'train_passes':int(tr.sum()),'test_passes':int(te.sum()),
                      'acc3':float((a==y3).mean()),'acc5':float((b==y5).mean()),
                      'pass_mae':float(abs(p-yp).mean())})
        for idx,aa,bb,pp in zip(np.where(te)[0],a,b,p):
            predictions.append({'grind':int(g[idx]),'pass_id':keys[idx][1],
                                'fold':fold,'pred3':int(aa),'pred5':int(bb),'progress':float(pp)})
        print('buffered fold',fold,folds[-1],flush=True)
    pred = pd.DataFrame(predictions).sort_values(['grind','pass_id'])
    y3,y5,yp = labels(pred.grind.to_numpy())
    blocked = {'acc3':float((pred.pred3.to_numpy()==y3).mean()),
               'acc5':float((pred.pred5.to_numpy()==y5).mean()),
               'pass_mae':float(abs(pred.progress.to_numpy()-yp).mean()),
               'n_passes':len(pred),'n_stages':int(pred.grind.nunique())}
    # Simple appearance-only reference using mean intensity and fixed monotonicity.
    iso = IsotonicRegression(increasing=True,out_of_bounds='clip')
    iso.fit(X[fit,0],g[fit]/169.)
    simple = iso.predict(X[test,0])
    intensity = {'pass_mae':float(abs(simple-target).mean()),
                 'spearman':float(spearmanr(simple,target).statistic),
                 'feature':'pass mean grayscale intensity; monotonic fit on even stages'}
    pd.DataFrame(metrics).to_csv(OUT/'stage_cluster_intervals.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'buffered_rf_folds.csv',index=False)
    pred.to_csv(OUT/'buffered_rf_predictions.csv',index=False)
    pd.DataFrame({'grind':g[test],'pass_id':[k[1] for k in testkeys],
                  'target_progress':target,'ours_progress':op,'b3_progress':bp,
                  'ours3':op3,'b3_3':bp3,'ours5':op5,'b5_5':rf5}).to_csv(OUT/'paired_pass_predictions.csv',index=False)
    summary = {'bootstrap_unit':'whole grinding stage; micro-average ratios',
               'bootstrap_seed':20261030,'scope':'conditional on one specimen trajectory',
               'intervals':metrics,'frame_ours_mae':frame,'buffered_rf':blocked,
               'buffered_rf_folds':folds,'intensity_reference':intensity,
               'pass_mae_ours':float(abs(op-target).mean()),
               'pass_mae_b3':float(abs(bp-target).mean()),
               'frame_b3_paired_cache_available':False,
               'raw_image_example_exists':Path(meta.path.iloc[0]).exists(),
               'cnn_blocked_validation':'not performed; raw image files unavailable locally'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    main()
