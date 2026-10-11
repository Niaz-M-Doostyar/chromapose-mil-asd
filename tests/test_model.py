import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import ChromaPoseMIL, ChromaticPartDecomposition, count_params, mirror_clips
from palette import COLORS_RGB, PAIRS

SKEL = np.array([[340, 120], [340, 170], [300, 170], [280, 230], [270, 290], [380, 170],
                 [400, 230], [410, 290], [315, 300], [310, 380], [305, 460], [365, 300],
                 [370, 380], [375, 460], [330, 110], [350, 110], [320, 115], [360, 115]])


def render(kp, h=554, w=696):
    img = np.zeros((h, w, 3), np.uint8)
    for a, b in PAIRS:
        cv2.line(img, tuple(map(int, kp[a])), tuple(map(int, kp[b])),
                 COLORS_RGB[b].tolist(), 4, cv2.LINE_AA)
    for i, p in enumerate(kp):
        cv2.circle(img, tuple(map(int, p)), 4, COLORS_RGB[i].tolist(), -1, cv2.LINE_AA)
    return img


def downsample(img, size=(64, 48)):
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def test_cpd_assigns_limbs():
    small = downsample(render(SKEL))
    x = torch.from_numpy(small).float()[None, None] / 255.0
    p = ChromaticPartDecomposition(motion=False)(x)[0, :, 0]
    mass = p.sum((1, 2))

    xs, ys = (SKEL[3] + SKEL[4]) / 2 * np.array([64 / 696, 48 / 554])
    col = p[:, int(ys), int(xs)]
    print("channel mass:", mass.numpy().round(2), "forearm argmax:", int(col.argmax()))
    assert int(col.argmax()) in (3, 4)


def test_mirror():
    small = torch.from_numpy(downsample(render(SKEL)))[None, None]
    m = mirror_clips(small)
    cpd = ChromaticPartDecomposition(motion=False)
    a = cpd(small.float() / 255)[0, :, 0].sum((1, 2))
    b = cpd(m.float() / 255)[0, :, 0].sum((1, 2))
    print("r_arm mass before/after:", a[2:5].sum().item(), b[5:8].sum().item())
    assert abs(a[2:5].sum() - b[5:8].sum()) / a[2:5].sum() < 0.25


def test_speed():
    torch.set_grad_enabled(True)
    cfg = dict(input="cpd", learnable_palette=True, motion=True, width=[32, 48, 96, 160],
               dim=128, temporal="transformer", clip_len=32, dropout=0.1, aggregator="gated")
    dev = torch.device("mps")
    m = ChromaPoseMIL(cfg).to(dev)
    print("params:", count_params(m))
    x = torch.randint(0, 255, (8 * 12, 32, 48, 64, 3), dtype=torch.uint8, device=dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    for i in range(6):
        if i == 2:
            torch.mps.synchronize(); t = time.time()
        lg, cl, a, h = m(x, 12)
        loss = lg.mean() + cl.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize()
    print("train step (96 clips): %.3fs" % ((time.time() - t) / 4))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    test_cpd_assigns_limbs(); test_mirror(); test_speed()
