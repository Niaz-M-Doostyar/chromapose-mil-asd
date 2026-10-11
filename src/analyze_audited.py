
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from data import ROOT,stratified_folds
from metrics import binary_metrics,delong_ci,delong_test,mcnemar_test,bootstrap_ci

RUNS=ROOT/'results/runs'; OUT=ROOT/'results/audited'; OUT.mkdir(exist_ok=True)
LABELS={'audited_full':'ChromaPose-MIL','audited_rgb':'RGB plus motion','audited_fixedpal':'Fixed hue prototypes','audited_nomotion':'Without motion channels','audited_meanpool':'Mean session pooling','audited_notf':'Without temporal transformer','audited_nosplit':'Without split-bag terms','audited_noflip':'Without mirror augmentation','audited_naiveflip':'Spatial mirror only','audited_noclip':'Without auxiliary clip loss','audited_groups':'Six hue groups','cnn_lstm':'VGG16-LSTM adaptation','kinematic_lr':'Hue dynamics LR','kinematic_rf':'Hue dynamics RF','audited_layout_lr':'Static layout LR'}
LABELS['higherres_compact']='Higher-resolution compact model'

def planned(name, protocol, seed):
 if protocol == 'cv5':
  return seed == 0 or (name in ('audited_full','audited_layout_lr') and seed in (1,2))
 return (seed == 0 and name in ('audited_full','cnn_lstm','kinematic_lr',
                               'kinematic_rf','audited_layout_lr')) or name == 'audited_layout_lr'

def load_complete(name,protocol,seed=0):
 man=pd.read_csv(ROOT/'data_cache/manifest.csv').set_index('child_id')
 expected=set(man.index if protocol=='cv5' else man[man.split=='test'].index)
 files=list((RUNS/name).glob(f'{protocol}_s{seed}/preds.csv'))+list((RUNS/name).glob(f'{protocol}_s{seed}_f*/preds.csv'))
 if not files:return None
 df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
 if df.child_id.duplicated().any():raise ValueError(f'Duplicate predictions: {name}')
 if set(df.child_id)!=expected:return None
 if not np.isfinite(df.prob).all() or not df.prob.between(0,1).all():raise ValueError('Invalid probabilities')
 if not np.array_equal(df.label.values,man.loc[df.child_id,'label'].values):raise ValueError('Label mismatch')
 if protocol=='cv5':
  folds=stratified_folds(list(man.index),list(man.label),5,seed)
  for f,ids in enumerate(folds):
   if set(df[df.fold==f].child_id)!=set(ids):raise ValueError(f'Fold mismatch: {name}')
 return df.set_index('child_id').sort_index()

def holm(p):
 p=np.asarray(p); order=np.argsort(p); adjusted=np.empty(len(p)); prev=0.
 for rank,i in enumerate(order):prev=max(prev,min(1.,(len(p)-rank)*p[i])); adjusted[i]=prev
 return adjusted

def main():
 status={}; allrows=[]; comparisons=[]
 for proto in ('cv5','official'):
  pooled={}
  for name,label in LABELS.items():
   for seed in (0,1,2):
    df=load_complete(name,proto,seed)
    status[f'{name}/{proto}/seed{seed}']='complete' if df is not None else ('pending' if planned(name,proto,seed) else 'not_scheduled')
    if df is None:continue
    m=binary_metrics(df.label,df.prob); auc,lo,hi=delong_ci(df.label.values,df.prob.values)
    al,ah=bootstrap_ci(df.label.values,df.prob.values,lambda y,p:np.mean((p>=.5)==y),n=2000)
    allrows.append(dict(model=name,name=label,protocol=proto,seed=seed,n=len(df),**m,auc_lo=lo,auc_hi=hi,acc_lo=al,acc_hi=ah,brier=float(np.mean((df.prob-df.label)**2))))
    if seed==0:pooled[name]=df
  if 'audited_full' in pooled:
   ref=pooled['audited_full']
   for name,df in pooled.items():
    if name=='audited_full':continue
    assert ref.index.equals(df.index)
    _,_,z,p=delong_test(ref.label.values,ref.prob.values,df.prob.values)
    b,c,pm=mcnemar_test(ref.label.values,ref.prob.values>=.5,df.prob.values>=.5)
    comparisons.append(dict(protocol=proto,model=name,auc_difference=binary_metrics(ref.label,ref.prob)['auc']-binary_metrics(df.label,df.prob)['auc'],delong_p=p,mcnemar_p=pm,discordant_ref_only=b,discordant_other_only=c))
  if pooled:
   fig,ax=plt.subplots(figsize=(3.45,3.1),dpi=600)
   for name,df in pooled.items():
    if name.startswith('audited_') and name not in ('audited_full','audited_layout_lr'):continue
    f,t,_=roc_curve(df.label,df.prob); auc=binary_metrics(df.label,df.prob)['auc']
    ax.plot(f,t,lw=1,label=f'{LABELS[name]} ({auc:.3f})')
   ax.plot([0,1],[0,1],':',color='gray',lw=.7); ax.set(xlabel='False positive rate',ylabel='Sensitivity'); ax.legend(fontsize=6,frameon=False); fig.tight_layout(); fig.savefig(OUT/f'roc_{proto}.png'); plt.close(fig)
  if 'audited_full' in pooled:
   m=binary_metrics(pooled['audited_full'].label,pooled['audited_full'].prob)
   cm=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
   fig,ax=plt.subplots(figsize=(3.2,2.7),dpi=600);ax.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max())
   for i in range(2):
    for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
   ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['TD','ASD'],yticklabels=['TD','ASD'],xlabel='Predicted',ylabel='Reference');fig.tight_layout();fig.savefig(OUT/f'confusion_{proto}.png');plt.close(fig)
 df=pd.DataFrame(allrows);df.to_csv(OUT/'metrics.csv',index=False)
 if comparisons:
  c=pd.DataFrame(comparisons)
  for proto,idx in c.groupby('protocol').groups.items():
   for key in ('delong_p','mcnemar_p'):c.loc[idx,key+'_holm']=holm(c.loc[idx,key].values)
  c.to_csv(OUT/'comparisons.csv',index=False)
 (OUT/'status.json').write_text(json.dumps(status,indent=2))
 print(df.to_string(index=False))
if __name__=='__main__':main()
