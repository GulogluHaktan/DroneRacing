"""Opponent-drone head on the GateNet decoder feature (one network, one trunk: no second detector).

GateNet.forward stores its last decoder feature ``net._y`` [B, 2w, H/2, W/2]. ``OppHead`` maps it to
4 channels: centre logit, (dx, dy) sub-cell offset in cells, ln(radius in input px). The trunk can stay frozen,
so the gate outputs of the trained GateNet are unchanged and only ~0.07 GFLOP are added.
Radius R_phys = 0.16 m (body + props) gives the distance:  d = f_px * 0.16 / r_px  (f_px of the full frame, r_px mapped back from the crop).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

R_PHYS = 0.16
STRIDE = 2


class OppHead(nn.Module):
    def __init__(self, c: int = 48, h: int = 32):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(c, h, 1), nn.ReLU(inplace=True), nn.Conv2d(h, h, 3, 1, 1), nn.ReLU(inplace=True),
                               nn.Conv2d(h, h, 3, 1, 2, dilation=2), nn.ReLU(inplace=True), nn.Conv2d(h, 4, 1))
        nn.init.constant_(self.a[-1].bias[:1], -3.0)

    def forward(self, y):
        return self.a(y)


def make_targets(c_px, r_px, valid, H, W, sigma=1.0):
    """c_px [B,M,2] centres, r_px [B,M] radii (input px), valid [B,M] -> heat [B,1,h,w], reg [B,3,h,w], mask [B,1,h,w]."""
    B, M = valid.shape
    h, w = H // STRIDE, W // STRIDE
    dev = c_px.device
    heat = torch.zeros(B, 1, h, w, device=dev)
    reg = torch.zeros(B, 3, h, w, device=dev)
    mask = torch.zeros(B, 1, h, w, device=dev)
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32), torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    for m in range(M):
        cc = c_px[:, m] / STRIDE - 0.5  # cell coordinates of the centre
        ix, iy = cc[:, 0].round().long(), cc[:, 1].round().long()
        ok = valid[:, m] & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        g = torch.exp(-((xs[None] - cc[:, 0, None, None]) ** 2 + (ys[None] - cc[:, 1, None, None]) ** 2) / (2 * sigma**2))
        g = g * ok[:, None, None].float()
        heat[:, 0] = torch.maximum(heat[:, 0], g)
        b = torch.nonzero(ok).flatten()
        if len(b):
            heat[b, 0, iy[b], ix[b]] = 1.0
            mask[b, 0, iy[b], ix[b]] = 1.0
            reg[b, 0, iy[b], ix[b]] = cc[b, 0] - ix[b]
            reg[b, 1, iy[b], ix[b]] = cc[b, 1] - iy[b]
            reg[b, 2, iy[b], ix[b]] = torch.log(r_px[b, m].clamp_min(0.5))
    return heat, reg, mask


@torch.no_grad()
def decode(o, thr=0.3, k=3):
    """o [B,4,h,w] -> list of [n,4] (score, u, v, r) in input px, at most k per frame (3x3 non-maximum suppression)."""
    p = torch.sigmoid(o[:, 0])
    pk = (p == F.max_pool2d(p[:, None], 3, 1, 1)[:, 0]) & (p > thr)
    res = []
    for b in range(o.shape[0]):
        iy, ix = torch.nonzero(pk[b], as_tuple=True)
        sc = p[b, iy, ix]
        top = sc.argsort(descending=True)[:k]
        iy, ix, sc = iy[top], ix[top], sc[top]
        u = (ix.float() + 0.5 + o[b, 1, iy, ix]) * STRIDE
        v = (iy.float() + 0.5 + o[b, 2, iy, ix]) * STRIDE
        r = torch.exp(o[b, 3, iy, ix].clamp(-3, 5))
        res.append(torch.stack((sc, u, v, r), -1))
    return res
