

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from palette import COLORS_RGB, MIRROR


def opponent_hue(rgb: torch.Tensor):

    r, g, b = rgb.unbind(-1)
    a = r - 0.5 * (g + b)
    bb = (math.sqrt(3.0) / 2.0) * (g - b)
    return torch.atan2(bb, a), torch.sqrt(a * a + bb * bb + 1e-8)


class ChromaticPartDecomposition(nn.Module):








    def __init__(self, learnable=True, kappa=30.0, motion=True, groups=None):
        super().__init__()
        protos = torch.from_numpy(COLORS_RGB.astype(np.float32) / 255.0)
        mu, _ = opponent_hue(protos)
        self.mu = nn.Parameter(mu.clone(), requires_grad=learnable)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(kappa)), requires_grad=learnable)
        self.motion = motion
        self.groups = groups
        k = len(groups) if groups else len(COLORS_RGB)
        self.out_channels = k * (2 if motion else 1)

    def forward(self, x):


        shape = x.shape[:-1]
        flat = x.reshape(-1, 3)
        idx = (flat.amax(-1) > 0.02).nonzero().squeeze(-1)
        hue, chroma = opponent_hue(flat[idx])
        sim = torch.cos(hue.unsqueeze(-1) - self.mu)
        w = torch.softmax(self.log_kappa.exp() * sim, dim=-1)
        v = chroma.unsqueeze(-1) * w
        if self.groups:
            v = torch.stack([v[:, g].sum(-1) for g in self.groups], -1)
        if torch.is_autocast_enabled(x.device.type):
            v = v.half()
        p = torch.zeros(flat.shape[0], v.shape[-1], device=x.device, dtype=v.dtype)
        p = p.index_copy(0, idx, v).view(*shape, -1)
        p = p.permute(0, 4, 1, 2, 3).contiguous()
        if self.motion:
            if p.shape[2] > 1:
                d = (p[:, :, 1:] - p[:, :, :-1]).abs()
                d = torch.cat([torch.zeros_like(d[:, :, :1]), d], 2)
            else:
                d = torch.zeros_like(p)
            p = torch.cat([p, d], 1)
        return p


class RGBInput(nn.Module):
    def __init__(self, motion=True):
        super().__init__()
        self.motion = motion
        self.out_channels = 6 if motion else 3

    def forward(self, x):
        p = x.permute(0, 4, 1, 2, 3).contiguous()
        if self.motion:
            if p.shape[2] > 1:
                d = (p[:, :, 1:] - p[:, :, :-1]).abs()
                p = torch.cat([p, torch.cat([torch.zeros_like(d[:, :, :1]), d], 2)], 1)
            else:
                p = torch.cat([p, torch.zeros_like(p)], 1)
        return p


class SpatialConv(nn.Module):


    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = x.transpose(1, 2).reshape(b * t, c, h, w)
        y = self.bn(self.conv(y))
        return y.view(b, t, *y.shape[1:]).transpose(1, 2)


class TemporalConv(nn.Module):


    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (k, 1), (stride, 1), (k // 2, 0), bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = self.bn(self.conv(x.reshape(b, c, t, h * w)))
        return y.view(b, y.shape[1], y.shape[2], h, w)


class Block2p1D(nn.Module):


    def __init__(self, cin, cout, s_stride=1, t_stride=1):
        super().__init__()
        mid = (cin + cout) // 2
        self.sp = SpatialConv(cin, mid, 3, s_stride)
        self.tp = TemporalConv(mid, cout, 3, t_stride)
        self.s_stride, self.t_stride = s_stride, t_stride
        self.skip = None
        if cin != cout or s_stride != 1 or t_stride != 1:
            self.skip = SpatialConv(cin, cout, 1, 1)

    def forward(self, x):
        y = self.tp(F.gelu(self.sp(x)))
        r = x[:, :, ::self.t_stride, ::self.s_stride, ::self.s_stride]
        r = r if self.skip is None else self.skip(r)
        return F.gelu(y + r)


class ClipEncoder(nn.Module):
    def __init__(self, cin, width=(32, 48, 96, 160), dim=128, temporal="transformer",
                 depth=2, heads=4, t_len=32, dropout=0.1):
        super().__init__()
        w0, w1, w2, w3 = width
        self.stem = nn.Sequential(SpatialConv(cin, w0, 3, 2), nn.GELU())
        self.stages = nn.Sequential(
            Block2p1D(w0, w1, s_stride=2, t_stride=1),
            Block2p1D(w1, w2, s_stride=2, t_stride=2),
            Block2p1D(w2, w3, s_stride=2, t_stride=2))
        self.proj = nn.Linear(w3, dim)
        self.temporal = temporal
        t_tokens = t_len // 4
        if temporal == "transformer":
            self.pos = nn.Parameter(torch.zeros(1, t_tokens, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout,
                                               activation="gelu", batch_first=True,
                                               norm_first=True)
            self.tf = nn.TransformerEncoder(layer, num_layers=depth,
                                            enable_nested_tensor=False)
            self.attn_pool = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x, return_maps=False):
        f = self.stages(self.stem(x))
        tok = self.proj(f.mean((3, 4)).transpose(1, 2))
        if self.temporal == "transformer":
            tok = self.tf(tok + self.pos[:, :tok.shape[1]])
            a = torch.softmax(self.attn_pool(tok), 1)
            z = (a * tok).sum(1)
        else:
            z = tok.mean(1)
        z = self.norm(z)
        return (z, f) if return_maps else z


class GatedAttentionMIL(nn.Module):


    def __init__(self, dim, hidden=64, mode="gated"):
        super().__init__()
        self.mode = mode
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h, mask=None):
        if self.mode == "mean":
            a = torch.ones(h.shape[:2], device=h.device)
            if mask is not None:
                a = a * mask
            a = a / a.sum(1, keepdim=True)
        else:
            s = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
            if mask is not None:
                s = s.masked_fill(mask == 0, -1e4)
            a = torch.softmax(s, 1)
        return (a.unsqueeze(-1) * h).sum(1), a


class ChromaPoseMIL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg["input"] == "rgb":
            self.inp = RGBInput(motion=cfg["motion"])
        else:
            from palette import PART_GROUPS
            groups = list(PART_GROUPS.values()) if cfg["input"] == "cpd_groups" else None
            self.inp = ChromaticPartDecomposition(learnable=cfg["learnable_palette"],
                                                  motion=cfg["motion"], groups=groups)
        self.enc = ClipEncoder(self.inp.out_channels, width=tuple(cfg["width"]),
                               dim=cfg["dim"], temporal=cfg["temporal"],
                               t_len=cfg["clip_len"], dropout=cfg["dropout"])
        self.mil = GatedAttentionMIL(cfg["dim"], mode=cfg["aggregator"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(cfg["dim"], 1)
        self.clip_head = nn.Linear(cfg["dim"], 1)

    def encode(self, clips):
        if clips.dtype == torch.uint8:
            clips = clips.float() / 255.0
        return self.enc(self.inp(clips))

    def aggregate(self, h, mask=None):
        z, a = self.mil(h, mask)
        return self.head(self.drop(z)).squeeze(-1), a

    def forward(self, clips, n_per_bag):

        h = self.encode(clips)
        clip_logits = self.clip_head(h).squeeze(-1)
        h = h.view(-1, n_per_bag, h.shape[-1])
        logits, a = self.aggregate(h)
        return logits, clip_logits, a, h


def mirror_clips(x: torch.Tensor, semantic: bool = True) -> torch.Tensor:


    x = x.flip(-2)
    if not semantic:
        return x
    return recolor(x, MIRROR)


_LUT_CACHE = {}


def recolor(x, perm):

    key = (x.device.type, tuple(perm))
    if key not in _LUT_CACHE:
        protos = torch.from_numpy(COLORS_RGB.astype(np.float32) / 255.0).to(x.device)
        mu, _ = opponent_hue(protos)
        _LUT_CACHE[key] = (mu, protos[torch.tensor(perm, device=x.device)])
    mu, target = _LUT_CACHE[key]
    xf = x.float() / 255.0 if x.dtype == torch.uint8 else x
    hue, chroma = opponent_hue(xf)
    idx = torch.cos(hue.unsqueeze(-1) - mu).argmax(-1)
    vmax = xf.max(-1).values
    col = target[idx] * vmax.unsqueeze(-1)
    out = torch.where((chroma > 0.08).unsqueeze(-1), col, xf)
    return (out * 255).round().clamp(0, 255).to(torch.uint8) if x.dtype == torch.uint8 else out


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
