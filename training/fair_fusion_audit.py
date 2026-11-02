"""Fixed two-expert controls from saved CNN predictions and feature-cache RF.

Post-hoc diagnostic, with no test-based weight search. All progress metrics
in this comparison are pass-level; only the cls5 combination changes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from training.revision_stage_audit import forest_predict, labels, clustered_interval

ROOT=Path(__file__).resolve().parents[1]

def main():
    meta=pd.read_csv(ROOT/'outputs/baselines/features/frame_features_meta.csv')
    xf=np.load(ROOT/'outputs/baselines/features/frame_features.npz')['X']
    keys=[]; xs=[]
    for key,df in meta.groupby(['grind','pass_id'],sort=True):
        x=xf[df.index.to_numpy()]
        keys.append(tuple(map(int,key)))
        xs.append(np.concatenate([x.mean(0),x.std(0),x.max(0),x.min(0)]))
    X=np.asarray(xs,dtype=np.float32); g=np.array([k[0] for k in keys])
    tr=g%2==0; te=g%2==1
    r3,r5,rp,rprob=forest_predict(X,g,tr,te)
    keys=[k for k,t in zip(keys,te) if t]; gt=g[te]
    y3,y5,yp=labels(gt)
    cache=ROOT/'outputs/baselines/ours_nested/pack_cache'
    c=np.load(cache/'rn4ens_test.npy',allow_pickle=True).item()
    b=np.load(cache/'rn_single_test.npy',allow_pickle=True).item()
    c3=np.array([c[k]['p3'] for k in keys]).argmax(1)
    cprob=np.array([c[k]['p5'] for k in keys]); cp=np.array([c[k]['pp'] for k in keys])
    avg=((cprob+rprob)/2).argmax(1)
    mc=np.where(cprob.max(1)>=rprob.max(1),cprob.argmax(1),r5)
    variants=[('Single ResNet',np.array([b[k]['p3'] for k in keys]).argmax(1),
               np.array([b[k]['p5'] for k in keys]).argmax(1),np.array([b[k]['pp'] for k in keys])),
              ('CNN ensemble',c3,cprob.argmax(1),cp),
              ('Pooled RF',r3,r5,rp),
              ('CNN + RF average',c3,avg,cp),
              ('PASS-Ord maxconf',c3,mc,cp)]
    rows=[]
    for name,a,b,p in variants:
        rows.append({'variant':name,'cls3_acc':float((a==y3).mean()),
                     'cls5_acc':float((b==y5).mean()),'pass_mae':float(abs(p-yp).mean()),
                     'pass_spearman':float(spearmanr(p,yp).statistic)})
    assert np.isclose(rows[-1]['cls5_acc'],.981859410430839)
    intervals={name:clustered_interval((mc==y5).astype(float)-(pred==y5),gt)
               for name,pred in [('maxconf_vs_average',avg),('maxconf_vs_CNN',cprob.argmax(1))]}
    out=ROOT/'outputs/baselines/revision_audit'
    pd.DataFrame(rows).to_csv(out/'fair_fusion_controls.csv',index=False)
    (out/'fair_fusion_summary.json').write_text(json.dumps({'rows':rows,
        'stage_intervals':intervals,'weight':.5,'test_passes':len(gt),
        'scope':'post-hoc fixed controls on the previously examined test set; no tuning'},indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(9,3.1))
    for ax,col,title in zip(axes,['cls3_acc','cls5_acc','pass_mae'],
                           [r'(a) Acc$_3$',r'(b) Acc$_5$','(c) Pass MAE']):
        vals=np.array([r[col] for r in rows]); y=np.arange(len(rows))[::-1]
        ax.scatter(vals,y,c=['#777777','#56B4E9','#E69F00','#009E73','#0072B2'],s=35)
        for v,yy in zip(vals,y):ax.annotate(f'{v:.4f}',(v,yy),xytext=(5,3),textcoords='offset points',fontsize=7)
        ax.set_yticks(y);ax.set_yticklabels([r['variant'] for r in rows] if ax==axes[0] else [])
        ax.set_ylim(-.6,4.6);ax.set_title(title,fontweight='bold')
        width=max(np.ptp(vals),.005);ax.set_xlim(vals.min()-width*.2,vals.max()+width*.65)
        ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.2)
    fig.subplots_adjust(left=.18,right=.98,bottom=.16,top=.85,wspace=.32)
    fig.suptitle('Fixed CNN/RF controls: matched pass-level evaluation',fontsize=10)
    fig.savefig(ROOT/'OJIM_LaTeX_Template/figures/Fig04_ablation.pdf',bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'rows':rows,'stage_intervals':intervals},indent=2))

if __name__=='__main__':main()
