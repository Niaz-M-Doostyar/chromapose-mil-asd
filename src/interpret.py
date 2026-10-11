
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT, Cohort
from model import ChromaPoseMIL, ChromaticPartDecomposition
from palette import PART_GROUPS

FIG = ROOT / "results/figures"
PAP = ROOT / "paper/figures"


@torch.no_grad()
def part_occlusion(model, cohort, cid, chunk=64):

    starts = cohort.all_clips(cid)
    if len(starts) > 80:
        starts = starts[:: max(1, len(starts) // 80)]
    clips = cohort.gather(cid, starts)
    base, _ = model.aggregate(model.encode(clips)[None])
    p0 = torch.sigmoid(base).item()
    groups = list(PART_GROUPS.values())
    names = list(PART_GROUPS)
    cpd = model.inp
    drops = []
    x = clips.float() / 255.0
    parts = cpd(x)
    k = parts.shape[1] // (2 if cpd.motion else 1)
    for g in groups:
        z = parts.clone()
        for i in g:
            z[:, i] = 0
            if cpd.motion:
                z[:, i + k] = 0
        h = model.enc(z)[None]
        logit, _ = model.aggregate(h)
        drops.append(p0 - torch.sigmoid(logit).item())
    return p0, dict(zip(names, drops))


def attention_figure(preds, cohort, path, n=6):
    ids = list(preds.keys())
    labs = np.array([cohort.label[c] for c in ids])
    ps = np.array([preds[c]["prob"] for c in ids])
    pick = []
    for lab in (1, 0):
        idx = np.where(labs == lab)[0]
        order = idx[np.argsort(-np.abs(ps[idx] - 0.5))]
        pick += [ids[i] for i in order[: n // 2]]
    fig, ax = plt.subplots(len(pick), 1, figsize=(7.16, 0.85 * len(pick) + 0.4), dpi=300,
                           sharex=True)
    if len(pick) == 1:
        ax = [ax]
    for a, c in zip(ax, pick):
        d = preds[c]
        t = d["starts"] / (8 * 60)
        w = d["attn"]
        a.fill_between(t, w, color="#b94a48" if cohort.label[c] else "#4a6fa5", alpha=0.85)
        a.set_ylabel(c, fontsize=6)
        a.tick_params(labelsize=6)
        a.set_yticks([])
        a.set_title(f"{c}  P={d['prob']:.2f}  {'ASD' if cohort.label[c] else 'TD'}",
                    loc="left", fontsize=7)
    ax[-1].set_xlabel("Session time (min)")
    fig.tight_layout()
    fig.savefig(path); plt.close()


def occlusion_figure(rows, path):
    names = list(rows[0][2].keys())
    asd = np.array([r[2][n] for r in rows if r[1] == 1]).mean(0)
    td = np.array([r[2][n] for r in rows if r[1] == 0]).mean(0)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.16, 2.6), dpi=300)
    ax.bar(x - 0.18, asd, 0.36, label="ASD", color="#b94a48")
    ax.bar(x + 0.18, td, 0.36, label="TD", color="#4a6fa5")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel(r"$\Delta P$(ASD) when group is occluded")
    ax.legend(frameon=False, fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(path); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/runs/full/cv5_s0_f0")
    a = ap.parse_args()
    run = ROOT / a.run
    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ChromaPoseMIL(ckpt["cfg"]).to(dev)
    model.load_state_dict(ckpt["state"]); model.eval()
    cohort = Cohort(dev, clip_len=ckpt["cfg"]["clip_len"])
    preds = np.load(run / "test_details.npy", allow_pickle=True).item()
    FIG.mkdir(parents=True, exist_ok=True); PAP.mkdir(parents=True, exist_ok=True)
    attention_figure(preds, cohort, FIG / "attention_timelines.png")
    rows = []
    for c in list(preds)[:24]:
        p0, d = part_occlusion(model, cohort, c)
        rows.append((c, cohort.label[c], d, p0))
        print(c, p0, {k: round(v, 3) for k, v in d.items()}, flush=True)
    occlusion_figure(rows, FIG / "part_occlusion.png")
    for name in ["attention_timelines.png", "part_occlusion.png"]:
        src = FIG / name
        if src.exists():
            (PAP / name).write_bytes(src.read_bytes())


if __name__ == "__main__":
    main()
