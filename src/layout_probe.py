
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import ROOT, stratified_folds
from metrics import binary_metrics

def main():
 man=pd.read_csv(ROOT/'data_cache/manifest.csv').set_index('child_id')
 ids=list(man.index); labels=man.label.to_dict(); rows=[]
 for c in ids:
  z=np.load(ROOT/f'data_cache/frames/{c}.npz'); fr=z['frames']; occ=z['occupancy']
  mean=fr.max(-1).mean(0)
  rows.append(np.r_[mean.reshape(6,8,8,8).mean((1,3)).ravel()/255,occ.mean(),occ.std()])
 X=pd.DataFrame(rows,index=ids)
 X.to_csv(ROOT/'data_cache/layout_features.csv',index_label='child_id')
 for proto in ['cv5','official']:
  for seed in [0,1,2]:
   folds=stratified_folds(ids,[labels[c] for c in ids],5,seed)
   splits=[([c for c in ids if c not in f],f) for f in folds] if proto=='cv5' else [(list(man[man.split=='train'].index),list(man[man.split=='test'].index))]
   out=ROOT/f'results/runs/audited_layout_lr/{proto}_s{seed}'; out.mkdir(parents=True,exist_ok=True)
   preds=[]
   for fold,(tr,te) in enumerate(splits):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=5000,random_state=seed))
    model.fit(X.loc[tr], [labels[c] for c in tr])
    p=model.predict_proba(X.loc[te])[:,1]
    joblib.dump(model,out/f'fold{fold}.joblib')
    (out/f'split{fold}.json').write_text(json.dumps(dict(train=tr,test=te),indent=2))
    preds += [dict(child_id=c,label=labels[c],prob=float(v),seed=seed,fold=fold) for c,v in zip(te,p)]
   df=pd.DataFrame(preds); df.to_csv(out/'preds.csv',index=False)
   print(proto,seed,binary_metrics(df.label,df.prob),flush=True)
if __name__=='__main__': main()
