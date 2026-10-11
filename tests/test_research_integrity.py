
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data import stratified_folds,inner_split
from model import ChromaticPartDecomposition,RGBInput
from metrics import delong_ci,delong_test,mcnemar_test
from train import config_overrides

class IntegrityTests(unittest.TestCase):
 def test_override_string_parsing(self):
  cfg=config_overrides(['mirror=none','temporal=transformer','split_bag=false','width=[32,48,96,160]','epochs=40'])
  self.assertEqual(cfg['mirror'],'none');self.assertEqual(cfg['temporal'],'transformer')
  self.assertIs(cfg['split_bag'],False);self.assertEqual(cfg['epochs'],40)
  self.assertEqual(config_overrides(['mirror=naive'])['mirror'],'naive')
  with self.assertRaises(ValueError):config_overrides(['epochs=none'])
 def test_child_splits(self):
  ids=list(range(160)); labels=[i//80 for i in ids]
  folds=stratified_folds(ids,labels,5,0)
  self.assertEqual(sorted(sum(folds,[])),ids)
  for f in folds:
   self.assertEqual(len(f),32); self.assertEqual(sum(labels[i] for i in f),16)
   outer=[i for i in ids if i not in f]
   tr,va=inner_split(outer,[labels[i] for i in outer],.2,0)
   self.assertFalse(set(tr)&set(va)); self.assertFalse(set(f)&(set(tr)|set(va)))
 def test_single_and_empty_frame(self):
  for n in (1,4):
   x=torch.zeros(2,n,8,8,3)
   for layer in (ChromaticPartDecomposition(),RGBInput()):
    y=layer(x); self.assertEqual(y.shape[2],n); self.assertTrue(torch.isfinite(y).all())
 def test_paired_metrics(self):
  y=np.array([0,0,0,1,1,1]); p=np.array([.1,.2,.3,.7,.8,.9])
  self.assertEqual(delong_ci(y,p)[0],1)
  self.assertEqual(delong_test(y,p,p)[-1],1)
  self.assertEqual(mcnemar_test(y,p>.5,p>.5)[-1],1)
if __name__=='__main__': unittest.main()
