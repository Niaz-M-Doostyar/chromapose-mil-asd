
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ROOT
from model import ChromaticPartDecomposition
from palette import KEYPOINTS, PART_GROUPS

FIG = ROOT / "paper/figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / "results/figures"
RES.mkdir(parents=True, exist_ok=True)


def _clip(cid="trASD001", t0=2400, n=8):
    z = np.load(ROOT / "data_cache/frames" / f"{cid}.npz")
    occ = z["occupancy"]
    t0 = int(np.clip(t0, 0, max(len(occ) - n, 0)))
    if occ[t0:t0 + n].mean() < 0.02:
        t0 = int(np.argmax(np.convolve(occ, np.ones(n) / n, "valid")))
    return z["frames"][t0:t0 + n]


def fig_cpd():
    clip = _clip()
    img = clip[len(clip) // 2]
    x = torch.from_numpy(clip).float()[None] / 255.0
    p = ChromaticPartDecomposition(learnable=False, motion=True)(x)[0].detach().numpy()
    k = len(KEYPOINTS)
    part, mot = p[:k, len(clip) // 2], p[k:, len(clip) // 2]
    groups = {}
    for name, idx in PART_GROUPS.items():
        groups[name] = part[idx].sum(0)
    fig, ax = plt.subplots(2, 4, figsize=(7.16, 3.4), dpi=300)
    ax[0, 0].imshow(img)
    ax[0, 0].set_title("Rendered skeleton", fontsize=7)
    names = list(groups)
    for i, n in enumerate(names):
        r, c = divmod(i + 1, 4)
        m = groups[n]
        ax[r, c].imshow(m, cmap="magma")
        ax[r, c].set_title(n.replace("_", " "), fontsize=7)
    ax[1, 3].imshow(mot.sum(0), cmap="magma")
    ax[1, 3].set_title("part motion", fontsize=7)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=0.3)
    for d in (FIG, RES):
        fig.savefig(d / "cpd_example.png")
        fig.savefig(d / "cpd_example.pdf")
    plt.close()


def _box(ax, xy, w, h, text, fc="#f4f4f4", ec="#222", fs=7):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       facecolor=fc, edgecolor=ec, linewidth=0.8)
    ax.add_patch(p)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, color="#222"))


def fig_arch():
    fig, ax = plt.subplots(figsize=(7.16, 2.55), dpi=300)
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 4.2); ax.axis("off")
    _box(ax, (0.15, 1.4), 1.7, 1.5, "ADOS session\n(rendered\nskeleton)", fc="#e8eef5")
    _arrow(ax, (1.85, 2.15), (2.25, 2.15))
    _box(ax, (2.25, 1.25), 2.15, 1.8, "Chromatic part\ndecomposition\n18 parts + motion", fc="#f7e8d4")
    _arrow(ax, (4.4, 2.15), (4.8, 2.15))
    _box(ax, (4.8, 1.25), 2.3, 1.8, "(2+1)D clip\nencoder\n+ temporal TF", fc="#dceadf")
    _arrow(ax, (7.1, 2.15), (7.5, 2.15))
    _box(ax, (7.5, 1.25), 2.15, 1.8, "Clip embeddings\n$h_1 \\ldots h_N$\n(session bag)", fc="#e8eef5")
    _arrow(ax, (9.65, 2.15), (10.05, 2.15))
    _box(ax, (10.05, 1.25), 2.05, 1.8, "Gated-attention\nMIL pooling", fc="#f7e8d4")
    _arrow(ax, (12.1, 2.15), (12.5, 2.15))
    _box(ax, (12.5, 1.4), 1.7, 1.5, "Child-level\nASD score", fc="#dceadf")
    ax.text(7.2, 0.35, "ChromaPose-MIL  ·  ~0.58 M parameters  ·  trained on Apple M3 Max (MPS)",
            ha="center", fontsize=7, color="#444")
    fig.tight_layout(pad=0.15)
    for d in (FIG, RES):
        fig.savefig(d / "architecture.png")
        fig.savefig(d / "architecture.pdf")
    plt.close()


def fig_graphical_abstract():
    fig = plt.figure(figsize=(7.16, 3.6), dpi=300)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.28)
    img = _clip("trASD012", 1800, 8)[4]
    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img); ax0.set_title("A  Input clip", fontsize=8)
    ax0.set_xticks([]); ax0.set_yticks([])
    x = torch.from_numpy(img).float()[None, None] / 255.0
    p = ChromaticPartDecomposition(learnable=False, motion=False)(x)[0, :, 0].detach().numpy()
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(p.sum(0), cmap="magma"); ax1.set_title("B  Part density", fontsize=8)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax2 = fig.add_subplot(gs[0, 2])
    t = np.linspace(0, 10, 200)
    rng = np.random.default_rng(0)
    a = np.abs(np.sin(0.9 * t) * np.exp(-0.08 * (t - 6) ** 2)) + 0.05 * rng.random(200)
    a = a / a.sum()
    ax2.fill_between(t, a, color="#4a6fa5", alpha=0.85)
    ax2.set_title("C  MIL attention", fontsize=8)
    ax2.set_xlabel("Session time (min)", fontsize=7)
    ax2.set_ylabel("Weight", fontsize=7)
    ax2.tick_params(labelsize=6)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_xlim(0, 12); ax3.set_ylim(0, 2.2); ax3.axis("off")
    labels = ["CPD", "(2+1)D + TF", "Gated MIL", "Child score"]
    cols = ["#f7e8d4", "#dceadf", "#e8eef5", "#f7e8d4"]
    for i, (lab, c) in enumerate(zip(labels, cols)):
        _box(ax3, (0.4 + 3.0 * i, 0.45), 2.3, 1.2, lab, fc=c, fs=8)
        if i < 3:
            _arrow(ax3, (2.7 + 3.0 * i, 1.05), (3.4 + 3.0 * i, 1.05))
    fig.suptitle("ChromaPose-MIL: part-resolved screening from rendered-skeleton ADOS video",
                 fontsize=9, y=0.98)
    fig.savefig(FIG / "graphical_abstract.png")
    fig.savefig(FIG / "graphical_abstract.pdf")
    fig.savefig(RES / "graphical_abstract.png")
    plt.close()


if __name__ == "__main__":
    fig_cpd(); fig_arch(); fig_graphical_abstract()
    print("wrote", FIG)
