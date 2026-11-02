"""Recompute retained pass metrics. No model loading or training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

root=Path(__file__).resolve().parent
d=pd.read_csv(root/'evidence/predictions/paired_pass_predictions.csv')
g=d.grind.to_numpy()
y3=np.where(g<=40,0,np.where(g<=110,1,2)); y5=(g-1)//34
def score(a,b,c,x,y,z):
    return {'acc3':float(np.mean(np.asarray(a)==np.asarray(x))),
            'acc5':float(np.mean(np.asarray(b)==np.asarray(y))),
            'pass_mae':float(np.mean(abs(np.asarray(c)-np.asarray(z)))),
            'pass_spearman':float(spearmanr(c,z).statistic)}
result={'PASS-Ord':score(y3,y5,d.target_progress,d.ours3,d.ours5,d.ours_progress)}
c=pd.read_csv(root/'evidence/predictions/cnn_ensemble_pass_predictions.csv')
assert list(zip(c.grind,c.pass_id))==list(zip(d.grind,d.pass_id))
result['CNN ensemble']=score(y3,y5,d.target_progress,c.pred_cls3,c.pred_cls5,c.pred_progress)
for name,file in [('SVM','svm_pass_predictions.csv'),('Frame-feature RF','rf_pass_predictions.csv')]:
    q=pd.read_csv(root/'evidence/classical_nested'/file)
    result[name]=score(q.true_cls3,q.true_cls5,q.true_progress,q.pred_cls3,q.pred_cls5,q.pred_progress)
expected=json.loads((root/'evidence/recomputed_metrics.json').read_text())
for name,metrics in result.items():
    for k,v in metrics.items(): assert abs(v-expected[name][k])<1e-7,(name,k)
print(json.dumps(result,indent=2))
print('PASS: retained metrics agree within 1e-7. No model was trained.')
