







import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision

ROOT = Path(__file__).resolve().parents[1]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
KMEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
KSTD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


def decode(template, n_parts, fps=4, size=224, max_s=700):
    lst = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    lst.write("".join(f"file '{template.format(i)}'\n" for i in range(1, n_parts + 1)))
    lst.close()
    cmd = ["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst.name, "-t", str(max_s),
           "-vf", f"fps={fps},scale={size}:{size}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    Path(lst.name).unlink()
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)


@torch.no_grad()
def vgg16(man, out, dev):
    net = torchvision.models.vgg16(weights="IMAGENET1K_V1").features.to(dev).eval()
    for p, n, cid in zip(man.path, man.n_parts, man.child_id):
        f = out / f"{cid}.npy"
        if f.exists():
            continue
        fr = decode(p, n)
        feats = []
        for i in range(0, len(fr), 128):
            x = torch.from_numpy(fr[i:i + 128]).to(dev).permute(0, 3, 1, 2).float() / 255
            x = (x - MEAN.to(dev)) / STD.to(dev)
            feats.append(net(x).mean((2, 3)).half().cpu())
        np.save(f, torch.cat(feats).numpy())
        print("vgg16", cid, len(fr), flush=True)


@torch.no_grad()
def r2p1d(man, out, dev, clip=16, hop=16):
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
    net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    net.fc = torch.nn.Identity()
    net = net.to(dev).eval()
    for cid in man.child_id:
        f = out / f"{cid}.npy"
        if f.exists():
            continue
        fr = np.load(ROOT / "data_cache/frames" / f"{cid}.npz")["frames"][::2]
        starts = np.arange(0, max(len(fr) - clip, 0) + 1, hop)
        feats = []
        for i in range(0, len(starts), 16):
            idx = starts[i:i + 16, None] + np.arange(clip)[None]
            x = torch.from_numpy(fr[idx]).to(dev).float() / 255
            b, t, h, w, _ = x.shape
            x = x.reshape(b * t, h, w, 3).permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
            x = x.view(b, t, 3, 112, 112).permute(0, 2, 1, 3, 4)
            x = (x - KMEAN.to(dev)) / KSTD.to(dev)
            feats.append(net(x).half().cpu())
        np.save(f, torch.cat(feats).numpy())
        print("r2p1d", cid, len(starts), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["vgg16", "r2p1d"])
    a = ap.parse_args()
    man = pd.read_csv(ROOT / "data_cache/manifest.csv")
    out = ROOT / "data_cache" / f"feat_{a.which}"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps")
    (vgg16 if a.which == "vgg16" else r2p1d)(man, out, dev)
