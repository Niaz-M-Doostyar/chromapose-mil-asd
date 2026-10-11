
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from threadpoolctl import threadpool_limits
from data import ROOT, stratified_folds
from metrics import binary_metrics, bootstrap_ci, delong_ci


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--temporal',action='store_true');args=ap.parse_args()
 out=ROOT/'results'/('nested_temporal' if args.temporal else 'nested_features');out.mkdir(exist_ok=True)
 man=pd.read_csv(ROOT/'data_cache/manifest.csv');ids=man.child_id.tolist();y=man.label.to_numpy()
 kin=np.load(ROOT/'data_cache/handcrafted.npz',allow_pickle=True)['kin'].item()
 mean=[];rich=[];pyramid=[];multiscale=[]
 for c in ids:
  x=np.load(ROOT/f'data_cache/feat_vgg16/{c}.npy').astype(np.float32)
  mean.append(x.mean(0))
  rich.append(np.concatenate([x.mean(0),x.std(0),np.abs(np.diff(x,axis=0)).mean(0)]))
  if args.temporal:
   pyramid.append(np.concatenate([np.concatenate([z.mean(0),z.std(0)]) for n in [1,2,4] for z in np.array_split(x,n)]))
   multiscale.append(np.concatenate([x.mean(0),x.std(0)]+[np.abs(x[lag:]-x[:-lag]).mean(0) for lag in [1,4,16,64]]))
 features={'appearance_mean':np.array(mean),'appearance_dynamics':np.array(rich),'hue_dynamics':np.array([kin[c] for c in ids])}
 features['fusion']=np.concatenate([features['appearance_dynamics'],features['hue_dynamics']],axis=1)
 if args.temporal:
  features={'temporal_pyramid':np.array(pyramid),'multiscale_motion':np.array(multiscale),'appearance_dynamics':np.array(rich),'temporal_fusion':np.concatenate([np.array(multiscale),features['hue_dynamics']],axis=1)}
 candidates=[(f,kind,c) for f in features for kind in ['lr','rbf'] for c in [.01,.1,1,10]]
 protocol=dict(outer_folds=5,outer_seed=0,inner_folds=4,selection='accuracy',threshold='fixed classifier decision boundary',candidates=candidates,note='Exploratory follow-up after prior results; no independent external cohort. VGG16 is pretrained, not from scratch.')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 folds=stratified_folds(ids,y,5,0);rows=[]
 for k,test_ids in enumerate(folds):
  tr=np.array([i for i,c in enumerate(ids) if c not in test_ids]);te=np.array([ids.index(c) for c in test_ids])
  cv=list(StratifiedKFold(4,shuffle=True,random_state=991+k).split(tr,y[tr]))
  assert not set(tr)&set(te)
  for a,b in cv:
   assert not set(tr[a])&set(tr[b])
   assert not (set(tr[a])|set(tr[b]))&set(te)
  scores=[]
  for f,kind,c in candidates:
   model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000) if kind=='lr' else SVC(C=c,kernel='rbf'))
   with threadpool_limits(limits=2):
    s=cross_val_score(model,features[f][tr],y[tr],cv=cv,scoring='accuracy',n_jobs=1)
   scores.append(float(s.mean()))
  best=int(np.argmax(scores));f,kind,c=candidates[best]
  model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000) if kind=='lr' else SVC(C=c,kernel='rbf'))
  with threadpool_limits(limits=2):model.fit(features[f][tr],y[tr])
  decision=model.decision_function(features[f][te]);prob=1/(1+np.exp(-decision))

  for i,p,d in zip(te,prob,decision):rows.append(dict(child_id=ids[i],label=int(y[i]),prob=float(p),decision=float(d),fold=k))
  joblib.dump(model,out/f'fold{k}.joblib')
  (out/f'fold{k}.json').write_text(json.dumps(dict(train=[ids[i] for i in tr],test=test_ids,inner_splits=[dict(train=[ids[i] for i in tr[a]],validation=[ids[i] for i in tr[b]]) for a,b in cv],selected=candidates[best],inner_accuracy=scores[best],candidate_scores=scores),indent=2))
  print(f'Fold {k}: selected {candidates[best]}, inner accuracy {scores[best]:.3f}',flush=True)
 assert len(rows)==len(ids) and len({r['child_id'] for r in rows})==len(ids)
 pd.DataFrame(rows).to_csv(out/'predictions.csv',index=False)
 result=binary_metrics(np.array([r['label'] for r in rows]),np.array([r['prob'] for r in rows]))
 yy=np.array([r['label'] for r in rows]);pp=np.array([r['prob'] for r in rows])
 result['accuracy_ci']=bootstrap_ci(yy,pp,lambda y,p:binary_metrics(y,p)['acc']).tolist()
 result['auc_ci']=list(delong_ci(yy,pp)[1:])
 result['interpretation']='Exploratory nested CV. Pooled AUC combines differently scaled LR/SVM margins; accuracy is the primary endpoint. Confidence intervals condition on predictions and omit training-selection variability.'
 (out/'metrics.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
if __name__=='__main__':main()
