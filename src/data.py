
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import mirror_clips

ROOT = Path(__file__).resolve().parents[1]


class Cohort:


    def __init__(self, device, frames_dir=ROOT / "data_cache/frames",
                 manifest=ROOT / "data_cache/manifest.csv", clip_len=32, hop=16,
                 min_valid=0.5):
        self.man = pd.read_csv(manifest).set_index("child_id")
        self.ids = list(self.man.index)
        self.label = {c: int(self.man.loc[c, "label"]) for c in self.ids}
        chunks, self.offset, self.length, self.starts = [], {}, {}, {}
        pos = 0
        for c in self.ids:
            z = np.load(Path(frames_dir) / f"{c}.npz")
            fr, occ = z["frames"], z["occupancy"]
            chunks.append(torch.from_numpy(fr))
            self.offset[c], self.length[c] = pos, len(fr)
            pos += len(fr)
            nonempty = torch.from_numpy((occ > 1e-4).astype(np.float32))
            cand = np.arange(0, max(len(fr) - clip_len, 0) + 1, hop)
            frac = np.array([nonempty[s:s + clip_len].mean().item() for s in cand])
            keep = cand[frac >= min_valid]
            self.starts[c] = keep if len(keep) >= 8 else cand


        self.frames = torch.cat(chunks)
        del chunks
        self.device, self.clip_len = device, clip_len

    def gather(self, cid, starts):
        idx = self.offset[cid] + torch.as_tensor(np.asarray(starts))[:, None] \
            + torch.arange(self.clip_len)[None]
        idx = idx.clamp(max=self.offset[cid] + self.length[cid] - 1)
        return self.frames[idx].to(self.device, non_blocking=True)

    def sample_bag(self, cid, n, rng, jitter=8):
        s = self.starts[cid]
        pick = np.sort(rng.choice(s, size=n, replace=len(s) < n))
        pick = np.clip(pick + rng.integers(-jitter, jitter + 1, size=n), 0,
                       self.length[cid] - self.clip_len)
        return self.gather(cid, pick), pick

    def all_clips(self, cid):
        return self.starts[cid]


def spatial_augment(x, rng, scale=(0.85, 1.0), shift=0.08):

    m, t, h, w, _ = x.shape
    s = torch.from_numpy(rng.uniform(*scale, size=m).astype(np.float32)).to(x.device)
    tx = torch.from_numpy(rng.uniform(-shift, shift, size=m).astype(np.float32)).to(x.device)
    ty = torch.from_numpy(rng.uniform(-shift, shift, size=m).astype(np.float32)).to(x.device)
    theta = torch.zeros(m, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = s, s, tx, ty
    grid = F.affine_grid(theta, (m, 3, h, w), align_corners=False)
    grid = grid.repeat_interleave(t, 0)
    xf = x.reshape(m * t, h, w, 3).permute(0, 3, 1, 2).float()
    out = F.grid_sample(xf, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.permute(0, 2, 3, 1).reshape(m, t, h, w, 3).round().clamp(0, 255).to(torch.uint8)


def augment(x, rng, cfg):
    if cfg.get("aug_spatial", True):
        x = spatial_augment(x, rng)
    mode = cfg.get("mirror", "semantic")
    if mode != "none":
        flip = torch.from_numpy(rng.random(x.shape[0]) < 0.5).to(x.device)
        if flip.any():
            x = x.clone()
            x[flip] = mirror_clips(x[flip], semantic=(mode == "semantic"))
    return x


def stratified_folds(ids, labels, k, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        members = [i for i, l in zip(ids, labels) if l == cls]
        rng.shuffle(members)
        for j, m in enumerate(members):
            folds[j % k].append(m)
    return folds


def inner_split(ids, labels, frac, seed):
    rng = np.random.default_rng(seed + 991)
    val = []
    for cls in (0, 1):
        members = [i for i, l in zip(ids, labels) if l == cls]
        rng.shuffle(members)
        val += members[:max(1, math.ceil(frac * len(members)))]
    return [i for i in ids if i not in val], val
