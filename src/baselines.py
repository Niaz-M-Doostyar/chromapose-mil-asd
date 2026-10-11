










import argparse
import copy
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import ROOT, inner_split, stratified_folds
from metrics import binary_metrics
from model import ChromaticPartDecomposition, GatedAttentionMIL
from palette import PART_GROUPS

DEV = torch.device("mps")


def splits(man, protocol, seed):
    if protocol == "official":
        return [(list(man[man.split == "train"].child_id), list(man[man.split == "test"].child_id))]
    sub = man if protocol == "cv5" else man[man.split == "train"]
    ids, labels = list(sub.child_id), list(sub.label)
    folds = stratified_folds(ids, labels, 5, seed)
    return [([i for j, f in enumerate(folds) if j != k for i in f], folds[k]) for k in range(5)]



class SegLSTM(nn.Module):
    def __init__(self, d=512, h=256):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.5),
                                nn.Linear(256, 1))

    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1]).squeeze(-1)


class FeatMIL(nn.Module):
    def __init__(self, d=512, h=128):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Dropout(0.3))
        self.mil = GatedAttentionMIL(h)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        z, a = self.mil(self.proj(x))
        return self.head(z).squeeze(-1), a


def load_feats(kind, ids):
    return {c: torch.from_numpy(np.load(ROOT / f"data_cache/feat_{kind}/{c}.npy").astype(np.float32))
            for c in ids}


def seq_run(kind, feats, lab, tr, te, seed, epochs=40, out=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr_ids, va_ids = inner_split(tr, [lab[c] for c in tr], 0.2, seed)
    if kind == "cnn_lstm":
        net, seg = SegLSTM().to(DEV), 20
        segs = {c: feats[c][: len(feats[c]) // seg * seg].view(-1, seg, 512) for c in feats}

        def child_prob(c):
            return torch.sigmoid(net(segs[c].to(DEV))).mean().item()
    else:
        net = FeatMIL().to(DEV)

        def child_prob(c):
            return torch.sigmoid(net(feats[c][None].to(DEV))[0]).item()
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.01)
    best, state = 1e9, None
    history=[]
    for ep in range(epochs):
        net.train()
        order = list(rng.permutation(tr_ids))
        for b in range(0, len(order), 8):
            kids = order[b:b + 8]
            y = torch.tensor([lab[c] for c in kids], dtype=torch.float32, device=DEV)
            if kind == "cnn_lstm":
                xs = [segs[c][rng.choice(len(segs[c]), 16)] for c in kids]
                logit = net(torch.cat(xs).to(DEV))
                loss = F.binary_cross_entropy_with_logits(logit, y.repeat_interleave(16))
            else:
                xs = torch.stack([feats[c][np.sort(rng.choice(len(feats[c]), 12))] for c in kids])
                logit, _ = net(xs.to(DEV))
                loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = np.clip([child_prob(c) for c in va_ids], 1e-4, 1 - 1e-4)
            yv = np.array([lab[c] for c in va_ids])
            vl = -np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
        history.append(dict(epoch=ep+1,val_loss=float(vl)))
        if vl < best:
            best, state = vl, copy.deepcopy(net.state_dict())
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        torch.save(dict(state=state,kind=kind,seed=seed,epochs=epochs),out/'model.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2))
        (out/'split.json').write_text(json.dumps(dict(train=tr_ids,validation=va_ids,test=te),indent=2))
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return {c: child_prob(c) for c in te}



@torch.no_grad()
def kinematic_features(ids, fps=8):
    cpd = ChromaticPartDecomposition(learnable=False, motion=False,
                                     groups=list(PART_GROUPS.values())).to(DEV)
    rows, layout = {}, {}
    freqs = np.fft.rfftfreq(256, 1 / fps)
    bands = [(0.25, 1.0), (1.0, 2.0), (2.0, 4.0)]
    for c in ids:
        z = np.load(ROOT / f"data_cache/frames/{c}.npz")
        fr, occ = z["frames"], z["occupancy"]
        masses = []
        for i in range(0, len(fr), 512):
            x = torch.from_numpy(fr[i:i + 512]).to(DEV).float()[None] / 255
            masses.append(cpd(x)[0].sum((2, 3)).T.cpu())
        m = torch.cat(masses).numpy()
        e = np.abs(np.diff(m, axis=0))
        f = []
        for g in range(m.shape[1]):
            spec = np.abs(np.fft.rfft(e[: len(e) // 256 * 256, g].reshape(-1, 256), axis=1)) ** 2
            spec = spec.mean(0); tot = spec[1:].sum() + 1e-9
            f += [m[:, g].mean(), m[:, g].std(), e[:, g].mean(), e[:, g].std(),
                  np.percentile(e[:, g], 90)]
            f += [spec[(freqs >= lo) & (freqs < hi)].sum() / tot for lo, hi in bands]
        f += [occ.mean(), occ.std(), (occ < 1e-4).mean()]
        rows[c] = np.array(f, np.float32)
        g = fr.astype(np.float32).max(-1).mean(0)
        layout[c] = np.concatenate([g.reshape(6, 8, 8, 8).mean((1, 3)).ravel() / 255,
                                    [occ.mean(), occ.std()]])
    return rows, layout


def sk_run(model, X, lab, tr, te):
    Xtr = np.stack([X[c] for c in tr]); ytr = [lab[c] for c in tr]
    model.fit(Xtr, ytr)
    return dict(zip(te, model.predict_proba(np.stack([X[c] for c in te]))[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["cnn_lstm", "r2p1d_mil", "kinematic_lr",
                                                   "kinematic_rf", "layout_lr"])
    ap.add_argument("--protocols", nargs="+", default=["cv5", "official"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--name", default="baselines")
    a = ap.parse_args()
    man = pd.read_csv(ROOT / "data_cache/manifest.csv")
    lab = dict(zip(man.child_id, man.label))
    ids = list(man.child_id)
    kin = lay = None
    if any(w.startswith(("kinematic", "layout")) for w in a.which):
        cache = ROOT / "data_cache/handcrafted.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True); kin, lay = z["kin"].item(), z["lay"].item()
        else:
            kin, lay = kinematic_features(ids)
            np.savez(cache, kin=kin, lay=lay)
    for w in a.which:
        feats = load_feats("vgg16" if w == "cnn_lstm" else "r2p1d", ids) if w in ("cnn_lstm", "r2p1d_mil") else None
        for proto in a.protocols:
            for seed in a.seeds:
                out = ROOT / f"results/runs/{w}/{proto}_s{seed}"
                if (out / "preds.csv").exists():
                    continue
                out.mkdir(parents=True,exist_ok=True)
                (out/"protocol.json").write_text(json.dumps(dict(model=w,protocol=proto,seed=seed,neural_epochs=40),indent=2))
                rows = []
                for k, (tr, te) in enumerate(splits(man, proto, seed)):
                    if feats is not None:
                        p = seq_run(w, feats, lab, tr, te, seed, out=out/f"fold{k}")
                    elif w == "kinematic_lr":
                        p = sk_run(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)), kin, lab, tr, te)
                    elif w == "kinematic_rf":
                        p = sk_run(RandomForestClassifier(500, min_samples_leaf=2, random_state=seed), kin, lab, tr, te)
                    else:
                        p = sk_run(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)), lay, lab, tr, te)
                    (out/f"outer_split{k}.json").write_text(json.dumps(dict(train=tr,test=te),indent=2))
                    rows += [dict(child_id=c, label=lab[c], prob=float(p[c]), seed=seed, fold=k) for c in te]
                df = pd.DataFrame(rows); out.mkdir(parents=True, exist_ok=True)
                df.to_csv(out / "preds.csv", index=False)
                m = binary_metrics(df.label, df.prob)
                print(f"{w:13s} {proto:8s} s{seed} AUC {m['auc']:.3f} ACC {m['acc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
