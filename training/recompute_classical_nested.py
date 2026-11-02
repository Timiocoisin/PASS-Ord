"""New classical baselines; historical results remain untouched."""
from pathlib import Path
import hashlib, json, platform
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from training.features import load_feature_cache
from training.temporal import nested_part_from_grind
from training.labels import cls3, cls5, progress

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/baselines/revision_audit/classical_nested'

def aggregate(meta, values):
    d=meta[['grind','pass_id']].reset_index(drop=True).copy()
    a=np.asarray(values)
    if a.ndim==1: a=a[:,None]
    for j in range(a.shape[1]): d[f'p{j}']=a[:,j]
    return d.groupby(['grind','pass_id'],sort=True).mean()

def prediction(model, x, meta, task):
    if task=='progress': return aggregate(meta,model.predict(x))['p0'].to_numpy()
    # Mean decision scores (SVM) or probabilities (RF), then pass argmax.
    a=model.decision_function(x) if hasattr(model,'decision_function') else model.predict_proba(x)
    return model.classes_[aggregate(meta,a).to_numpy().argmax(1)]

def score(y,p,task):
    return float(mean_absolute_error(y,p) if task=='progress' else accuracy_score(y,p))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    x,m,names=load_feature_cache(ROOT/'outputs/baselines/features')
    assert np.isfinite(x).all()
    tags=m.grind.map(nested_part_from_grind).to_numpy()
    targets={'cls3':m.grind.map(cls3).to_numpy(),'cls5':m.grind.map(cls5).to_numpy(),
             'progress':m.grind.map(progress).to_numpy()}
    masks={k:tags==k for k in ['train','val','test']}
    masks['refit']=tags!='test'
    config={'date':'2026-10-30','python':platform.python_version(),'sklearn':sklearn.__version__,
      'features':names,'feature_level':'20D frame features',
      'protocol':'g%4=0 train; g%4=2 val; odd test; final refit all even',
      'aggregation':'mean decision scores (SVM) / probabilities (RF); mean progress per pass',
      'svm_grid':{'C':[0.1,1.,10.],'kernel':'rbf','gamma':'scale','epsilon_progress':0.01},
      'rf':{'n_estimators':400,'max_depth':16,'random_state':42},
      'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'outputs/baselines/features/frame_features.npz',ROOT/'outputs/baselines/features/frame_features_meta.csv']},
      'selection':{},'counts':{k:{'frames':int(v.sum()),'passes':len(m.loc[v,['grind','pass_id']].drop_duplicates()),'stages':int(m.loc[v,'grind'].nunique())} for k,v in masks.items()}}
    # Freeze selection on validation before evaluating any test predictions.
    for task,y in targets.items():
        va=m.loc[masks['val']].reset_index(drop=True)
        truth=va.groupby(['grind','pass_id'],sort=True).grind.first().map({'cls3':cls3,'cls5':cls5,'progress':progress}[task]).to_numpy()
        trials=[]
        for c in config['svm_grid']['C']:
            est=SVR(C=c,epsilon=.01) if task=='progress' else SVC(C=c,decision_function_shape='ovr')
            model=make_pipeline(StandardScaler(),est).fit(x[masks['train']],y[masks['train']])
            pred=prediction(model,x[masks['val']],va,task)
            trials.append({'C':c,'validation_score':score(truth,pred,task)})
        best=min(trials,key=lambda z:z['validation_score']) if task=='progress' else max(trials,key=lambda z:z['validation_score'])
        config['selection'][task]={'trials':trials,'selected_C':best['C']}
    (OUT/'frozen_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    test=m.loc[masks['test']].reset_index(drop=True)
    keys=test.groupby(['grind','pass_id'],sort=True).size().reset_index(name='n_frames')
    rows=[]
    for method in ['SVM','RF']:
        predictions=keys.copy(); metrics={'model':method,'progress_level':'pass','note':'new nested frame-feature baseline; pass-aggregated predictions'}
        for task,y in targets.items():
            if method=='SVM':
                c=config['selection'][task]['selected_C']
                est=SVR(C=c,epsilon=.01) if task=='progress' else SVC(C=c,decision_function_shape='ovr')
                model=make_pipeline(StandardScaler(),est)
            else:
                cls=RandomForestRegressor if task=='progress' else RandomForestClassifier
                model=cls(n_estimators=400,max_depth=16,random_state=42,n_jobs=-1)
            model.fit(x[masks['refit']],y[masks['refit']])
            p=prediction(model,x[masks['test']],test,task)
            truth=keys.grind.map({'cls3':cls3,'cls5':cls5,'progress':progress}[task]).to_numpy()
            predictions['true_'+task]=truth; predictions['pred_'+task]=p
            if task=='progress': metrics.update(progress_mae=score(truth,p,task),progress_spearman=float(spearmanr(truth,p).statistic))
            else: metrics.update({task+'_acc':score(truth,p,task),task+'_f1':float(f1_score(truth,p,average='macro'))})
        assert len(predictions)==441 and predictions.grind.nunique()==85
        predictions.to_csv(OUT/(method.lower()+'_pass_predictions.csv'),index=False)
        rows.append(metrics); print(json.dumps(metrics),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__': main()
