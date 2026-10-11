
import argparse
import copy
import json
import time
import platform
import sys
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import ROOT, Cohort, augment, inner_split, stratified_folds
from metrics import binary_metrics
from model import ChromaPoseMIL, count_params

DEFAULT = dict(
    input="cpd", learnable_palette=True, motion=True, width=[32, 48, 96, 160], dim=128,
    temporal="transformer", clip_len=32, dropout=0.2, aggregator="gated",
    bag=12, batch=8, epochs=40, lr=1e-3, wd=0.05, warmup=3, label_smooth=0.05,
    w_clip=0.3, w_cons=0.5, split_bag=True, mirror="semantic", aug_spatial=True,
    ema=0.99, val_frac=0.2, eval_every=3, patience=8, eval_hop=1, bags_per_epoch=3, amp=True,
)


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)


def config_overrides(settings):

    cfg = copy.deepcopy(DEFAULT)
    for setting in settings:
        k, v = setting.split("=", 1)
        if k not in cfg:
            raise ValueError(f"Unknown configuration key: {k}")
        value = json.loads(v) if not isinstance(cfg[k], str) or v.startswith('"') else v
        expected = type(cfg[k])
        if type(value) is not expected and not (expected is float and type(value) is int):
            raise ValueError(f"{k} must have type {expected.__name__}")
        cfg[k] = value
    return cfg


def smooth(y, eps):
    return y * (1 - eps) + 0.5 * eps


@torch.no_grad()
def predict_children(model, cohort, ids, chunk=128, hop=1, amp=True):

    model.eval()
    out = {}
    for c in ids:
        starts = cohort.all_clips(c)[::hop]
        hs = []
        for i in range(0, len(starts), chunk):
            with torch.autocast("mps", dtype=torch.float16, enabled=amp):
                hs.append(model.encode(cohort.gather(c, starts[i:i + chunk])).float())
        h = torch.cat(hs)[None]
        logit, a = model.aggregate(h)
        clip_p = torch.sigmoid(model.clip_head(h[0]).squeeze(-1))
        out[c] = dict(prob=torch.sigmoid(logit).item(), attn=a[0].cpu().numpy(),
                      clip_prob=clip_p.cpu().numpy(), starts=np.asarray(starts))
    return out


def run(cfg, cohort, train_ids, test_ids, seed, out_dir=None, log=print):
    seed_all(seed)
    rng = np.random.default_rng(seed)
    dev = cohort.device
    labels = [cohort.label[c] for c in train_ids]
    if cfg["val_frac"] > 0:
        tr_ids, va_ids = inner_split(train_ids, labels, cfg["val_frac"], seed)
    else:
        tr_ids, va_ids = list(train_ids), []
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        (out_dir / "split.json").write_text(json.dumps(dict(train=tr_ids, validation=va_ids, test=test_ids, seed=seed), indent=2))
        (out_dir / "environment.json").write_text(json.dumps(dict(
            python=sys.version, torch=torch.__version__, numpy=np.__version__,
            platform=platform.platform(), device=str(dev),
            sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__), Path(__file__).with_name('model.py'), Path(__file__).with_name('data.py')]}), indent=2))
        if hasattr(cohort, 'metadata_paths'):
            (out_dir / 'cache_provenance.json').write_text(json.dumps(dict(
                parameters=cohort.cache_parameters,
                metadata_sha256={c:hashlib.sha256(Path(cohort.metadata_paths[c]).read_bytes()).hexdigest()
                                 for c in cohort.ids}), indent=2))
    assert not (set(tr_ids) & set(va_ids) or set(tr_ids) & set(test_ids) or set(va_ids) & set(test_ids))
    model = ChromaPoseMIL(cfg).to(dev)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    pos = [c for c in tr_ids if cohort.label[c] == 1]
    neg = [c for c in tr_ids if cohort.label[c] == 0]
    steps_per_epoch = max(1, cfg["bags_per_epoch"] * len(tr_ids) // cfg["batch"])
    total = steps_per_epoch * cfg["epochs"]
    warm = steps_per_epoch * cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s / total, 1.0))))
    hist, best, best_state, bad, step = [], (1e9, -1), None, 0, 0
    t0 = time.time()
    n = cfg["bag"]
    for ep in range(cfg["epochs"]):
        model.train()
        order = []
        for _ in range(cfg["bags_per_epoch"]):
            rng.shuffle(pos); rng.shuffle(neg)
            order += [c for pair in zip(pos, neg) for c in pair] + pos[len(neg):] + neg[len(pos):]
        ep_loss = []
        for b in range(steps_per_epoch):
            kids = order[b * cfg["batch"]:(b + 1) * cfg["batch"]]
            if len(kids) < 2:
                continue
            clips = torch.cat([cohort.sample_bag(c, n, rng)[0] for c in kids])
            clips = augment(clips, rng, cfg)
            y = torch.tensor([cohort.label[c] for c in kids], dtype=torch.float32, device=dev)
            ys = smooth(y, cfg["label_smooth"])
            with torch.autocast("mps", dtype=torch.float16, enabled=cfg["amp"]):
                h = model.encode(clips).float()
            hb = h.view(len(kids), n, -1)
            logit, _ = model.aggregate(hb)
            loss = F.binary_cross_entropy_with_logits(logit, ys)
            if cfg["w_clip"] > 0:
                cl = model.clip_head(h).squeeze(-1)
                loss = loss + cfg["w_clip"] * F.binary_cross_entropy_with_logits(
                    cl, ys.repeat_interleave(n))
            if cfg["split_bag"]:
                perm = torch.randperm(n, device=dev)
                l1, _ = model.aggregate(hb[:, perm[: n // 2]])
                l2, _ = model.aggregate(hb[:, perm[n // 2:]])
                sup = 0.5 * (F.binary_cross_entropy_with_logits(l1, ys)
                             + F.binary_cross_entropy_with_logits(l2, ys))
                p1, p2 = torch.sigmoid(l1), torch.sigmoid(l2)
                cons = (p1 * (torch.log(p1 + 1e-6) - torch.log(p2 + 1e-6))
                        + (1 - p1) * (torch.log(1 - p1 + 1e-6) - torch.log(1 - p2 + 1e-6))
                        + p2 * (torch.log(p2 + 1e-6) - torch.log(p1 + 1e-6))
                        + (1 - p2) * (torch.log(1 - p2 + 1e-6) - torch.log(1 - p1 + 1e-6))).mean() / 2
                loss = loss + 0.5 * sup + cfg["w_cons"] * cons
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step(); sched.step(); step += 1
            with torch.no_grad():
                d = min(cfg["ema"], (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            ep_loss.append(loss.item())
        rec = dict(epoch=ep + 1, train_loss=float(np.mean(ep_loss)), lr=sched.get_last_lr()[0],
                   time=time.time() - t0)
        if not va_ids:
            best_state = ema.state_dict()
            if (ep + 1) % cfg["eval_every"] == 0:
                log(f"  ep{ep + 1:3d} loss {rec['train_loss']:.3f} ({rec['time']:.0f}s, "
                    f"mps {torch.mps.driver_allocated_memory() / 2**30:.1f} GB)")
        elif (ep + 1) % cfg["eval_every"] == 0 or ep + 1 == cfg["epochs"]:
            pv = predict_children(ema, cohort, va_ids, chunk=cfg.get('_eval_chunk',128), hop=2, amp=cfg["amp"])
            yv = np.array([cohort.label[c] for c in va_ids]); pr = np.array([pv[c]["prob"] for c in va_ids])
            vloss = float(F.binary_cross_entropy(torch.tensor(pr).clamp(1e-4, 1 - 1e-4),
                                                 torch.tensor(yv, dtype=torch.float64)))
            m = binary_metrics(yv, pr)
            torch.mps.empty_cache()
            rec.update(val_loss=vloss, val_auc=m["auc"], val_acc=m["acc"])
            key = (vloss, -m["auc"])
            if key < best:
                best, bad = key, 0
                best_state = copy.deepcopy(ema.state_dict()); rec["best"] = True
            else:
                bad += 1
            log(f"  ep{ep + 1:3d} loss {rec['train_loss']:.3f} val_loss {vloss:.3f} "
                f"val_auc {m['auc']:.3f} val_acc {m['acc']:.3f} ({rec['time']:.0f}s)")
            if bad >= cfg["patience"]:
                hist.append(rec); break
        hist.append(rec)
        if out_dir:
            (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
            if rec.get("best"):
                torch.save(dict(state=best_state, cfg=cfg), out_dir / "model.pt")
    ema.load_state_dict(best_state)
    preds = predict_children(ema, cohort, test_ids, chunk=cfg.get('_eval_chunk',128), hop=cfg["eval_hop"], amp=cfg["amp"])
    torch.mps.empty_cache()
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=best_state, cfg=cfg), out_dir / "model.pt")
        json.dump(hist, open(out_dir / "history.json", "w"), indent=1)
        np.save(out_dir / "test_details.npy", preds, allow_pickle=True)
    return preds, hist, count_params(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--protocol", choices=["official", "cv5", "dev"], default="cv5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="key=json overrides")
    ap.add_argument('--loader', choices=['compact','lazy'], default='compact')
    ap.add_argument('--frames-dir', type=Path, default=None)
    ap.add_argument('--eval-chunk', type=int, default=128)
    ap.add_argument('--require-mps', action='store_true')
    a = ap.parse_args()
    cfg = config_overrides(a.set)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if a.require_mps and dev.type != 'mps':
        raise RuntimeError('MPS is required for this experiment; no GPU training started')
    if a.eval_chunk < 1:
        raise ValueError('Evaluation chunk size must be positive')
    if a.loader == 'lazy':
        from lazy_cohort import LazyCohort
        frames_dir=(a.frames_dir or ROOT/'data_cache/frames128_compressed').resolve()
        cohort=LazyCohort(dev,frames_dir=frames_dir,clip_len=cfg['clip_len'])
        cfg.update(_loader='lazy',_frames_dir=str(frames_dir),_eval_chunk=a.eval_chunk)
    else:
        kwargs={'frames_dir':a.frames_dir} if a.frames_dir else {}
        cohort = Cohort(dev, clip_len=cfg["clip_len"],**kwargs)
        if a.frames_dir:cfg['_frames_dir']=str(a.frames_dir.resolve())
        if a.eval_chunk!=128:cfg['_eval_chunk']=a.eval_chunk
    base = ROOT / "results/runs" / a.name
    base.mkdir(parents=True, exist_ok=True)
    config_path = base / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError("Run name already has a different configuration; use a new name")
    config_path.write_text(json.dumps(cfg, indent=1))
    man = cohort.man
    rows = []
    for seed in a.seeds:
        if a.protocol == "official":
            splits = [(list(man[man.split == "train"].index), list(man[man.split == "test"].index))]
        else:
            ids = list(man.index) if a.protocol == "cv5" else list(man[man.split == "train"].index)
            labels = [cohort.label[c] for c in ids]
            folds = stratified_folds(ids, labels, 5, seed=seed)
            splits = [([i for j, f in enumerate(folds) if j != k for i in f], folds[k])
                      for k in range(5)]
        for k, (tr, te) in enumerate(splits):
            if a.folds is not None and k not in a.folds:
                continue
            tag = f"{a.protocol}_s{seed}_f{k}"
            if (base / tag / "preds.csv").exists():
                print("skip", tag); continue
            print(f"[{a.name}] {tag}: train {len(tr)} test {len(te)}", flush=True)
            preds, hist, npar = run(cfg, cohort, tr, te, seed, base / tag,
                                    log=lambda s: print(s, flush=True))
            import pandas as pd
            df = pd.DataFrame([dict(child_id=c, label=cohort.label[c], prob=preds[c]["prob"],
                                    prob_meanclip=float(preds[c]["clip_prob"].mean()),
                                    seed=seed, fold=k) for c in te])
            df.to_csv(base / tag / "preds.csv", index=False)
            m = binary_metrics(df.label, df.prob)
            print(f"[{a.name}] {tag} AUC {m['auc']:.3f} ACC {m['acc']:.3f} "
                  f"SENS {m['sensitivity']:.3f} SPEC {m['specificity']:.3f} params {npar}", flush=True)
            rows.append(df)


if __name__ == "__main__":
    main()
