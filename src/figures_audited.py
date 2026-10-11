
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from data import ROOT

OUT=ROOT/'results/audited';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
def save(fig,name):
 fig.savefig(OUT/f'{name}.png',dpi=600,bbox_inches='tight');fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def main():
 fig,ax=plt.subplots(figsize=(3.4,4.7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 boxes=[('Rendered ADOS skeleton videos','160 children; 80 ASD / 80 TD'),('Sampling and preprocessing','8 fps; 64 × 48 pixels; 4-second clips'),('Chromatic decomposition','18 hue channels + temporal differences'),('Clip encoder','Factorized convolutions + transformer'),('Session attention pooling','12 clips for training; full session at test'),('Child-level output','Research score and attention weights')]
 for i,(a,b) in enumerate(boxes):
  y=.87-i*.157
  ax.add_patch(FancyBboxPatch((.03,y),.94,.105,boxstyle='round,pad=.01',fc='#edf4f8',ec='#285b78',lw=.7))
  ax.text(.5,y+.069,a,ha='center',va='center',weight='bold',fontsize=8)
  ax.text(.5,y+.028,b,ha='center',va='center',fontsize=6.9)
  if i<5:ax.annotate('',xy=(.5,y-.046),xytext=(.5,y-.012),arrowprops={'arrowstyle':'->','lw':.8})
 save(fig,'architecture')
 hs=sorted((ROOT/'results/runs/audited_full').glob('cv5_s0_f*/history.json'))
 fig,axes=plt.subplots(2,1,figsize=(3.4,4.5))
 for p in hs:
  h=json.loads(p.read_text());k=p.parent.name.split('_f')[-1]
  axes[0].plot([r['epoch'] for r in h],[r['train_loss'] for r in h],label=f'Fold {int(k)+1}',lw=1)
  v=[r for r in h if 'val_auc' in r];axes[1].plot([r['epoch'] for r in v],[r['val_auc'] for r in v],lw=1)
 axes[0].set(ylabel='Training objective');axes[0].legend(fontsize=6,frameon=False,ncol=3)
 axes[1].set(xlabel='Epoch',ylabel='Inner-validation AUC',ylim=(0,1));fig.tight_layout();save(fig,'training')
 preds={}
 for p in sorted((ROOT/'results/runs/audited_full').glob('cv5_s0_f*/test_details.npy')):preds.update(np.load(p,allow_pickle=True).item())
 if preds:
  pick=[sorted(c for c in preds if 'ASD' in c)[0],sorted(c for c in preds if 'TD' in c)[0]]
  fig,axes=plt.subplots(2,1,figsize=(3.4,2.7),sharex=True)
  for ax,c in zip(axes,pick):
   d=preds[c];ax.plot(d['starts']/480,d['attn'],lw=.7);ax.set_title(f'{c}: score {d["prob"]:.3f}',fontsize=8,loc='left');ax.set_ylabel('Weight')
  axes[-1].set_xlabel('Time within concatenated release (min)');fig.tight_layout();save(fig,'attention')
  c=pick[0];z=np.load(ROOT/f'data_cache/frames/{c}.npz');frame=z['frames'][int(preds[c]['starts'][np.argmax(preds[c]['attn'])])]
  fig,axes=plt.subplots(1,3,figsize=(7.0,2.0),gridspec_kw={'width_ratios':[1,1.25,1.5]})
  axes[0].imshow(frame);axes[0].axis('off');axes[0].set_title('De-identified rendering',fontsize=8)
  axes[1].axis('off');axes[1].text(.5,.5,'Hue decomposition\n↓\nSpatiotemporal encoder\n↓\nSession pooling',ha='center',va='center',fontsize=9)
  d=preds[c];axes[2].plot(d['starts']/480,d['attn'],lw=.8);axes[2].set(xlabel='Session time (min)',ylabel='Attention weight');axes[2].set_title('Held-out model attention',fontsize=8);fig.tight_layout();save(fig,'graphical_abstract')
if __name__=='__main__':main()
