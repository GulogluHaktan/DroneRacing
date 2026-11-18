"""GateNet v2 helpers: corner heatmaps that do not depend on the gate centre being in the image.

``gatenet.make_corner_targets`` only labels corners of gates whose centre is inside the image and that show >= 2 corners,
so a gate that is close and partly out of view (centre outside) is never a training target: the detector is blind exactly
when the gate is nearest. Here every visible, in-image corner is a positive (gate needs >= 1 corner in the image), and the
decode returns corner peaks that are independent of the centre detection. Same network (``GateNet(refine=True)``), same
heads; only targets and decoding change, so ``gatenet.py`` and the g3 checkpoint stay valid.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def make_corner_targets_any(corners_px, vis, slot_valid, H, W, stride=2, sigma=1.5, min_in=1):
    """-> heat [B,4,h,w], sub-cell offsets [B,8,h,w], offset mask [B,8,h,w] for every visible in-image corner."""
    B, S = corners_px.shape[:2]
    h, w = H // stride, W // stride
    dev = corners_px.device
    sane = (corners_px.abs() < 4 * max(H, W)).all(-1).all(-1)
    cin = (corners_px[..., 0] >= 0) & (corners_px[..., 0] < W) & (corners_px[..., 1] >= 0) & (corners_px[..., 1] < H)
    ok_s = slot_valid & sane & (cin & vis).sum(-1).ge(min_in)
    pos = ok_s[..., None] & cin & vis  # [B,S,4]
    ys = torch.arange(h, device=dev).view(1, 1, 1, h, 1).float() + 0.5
    xs = torch.arange(w, device=dev).view(1, 1, 1, 1, w).float() + 0.5
    cxs = ((corners_px[..., 0] / stride).floor() + 0.5)[..., None, None]
    cys = ((corners_px[..., 1] / stride).floor() + 0.5)[..., None, None]
    heat = (torch.exp(-((xs - cxs) ** 2 + (ys - cys) ** 2) / (2 * sigma**2)) * pos[..., None, None]).amax(1)
    off = torch.zeros(B, 8, h, w, device=dev)
    mask = torch.zeros(B, 8, h, w, device=dev)
    ci = (corners_px[..., 0] / stride).long().clamp(0, w - 1)
    cj = (corners_px[..., 1] / stride).long().clamp(0, h - 1)
    frac = corners_px / stride - torch.stack((ci, cj), -1).float() - 0.5
    bi = torch.arange(B, device=dev)
    for s in range(S):
        for c in range(4):
            ok = pos[:, s, c]
            if ok.any():
                for k in (0, 1):
                    off[bi[ok], 2 * c + k, cj[ok, s, c], ci[ok, s, c]] = frac[ok, s, c, k]
                    mask[bi[ok], 2 * c + k, cj[ok, s, c], ci[ok, s, c]] = 1.0
    return heat, off, mask, pos


@torch.no_grad()
def decode_corner_peaks(ch_logits, ch_off, thr=0.3, k=8, stride=2):
    """Per image and corner type (TL, TR, BR, BL) the strongest heatmap peaks -> tensor [n, 4] rows (type, score, u, v) in pixels."""
    heat = torch.sigmoid(ch_logits)
    keep = (F.max_pool2d(heat, 3, 1, 1) == heat).float() * heat
    B, C, h, w = keep.shape
    out = []
    for b in range(B):
        rows = []
        for c in range(C):
            sc, idx = keep[b, c].view(-1).topk(k)
            m = sc > thr
            j, i = idx[m] // w, idx[m] % w
            u = (i.float() + 0.5 + ch_off[b, 2 * c, j, i]) * stride
            v = (j.float() + 0.5 + ch_off[b, 2 * c + 1, j, i]) * stride
            rows.append(torch.stack((torch.full_like(u, c), sc[m], u, v), -1))
        out.append(torch.cat(rows, 0))
    return out
