"""GateNet: gate detector + inner-corner regressor, trained from simulator ground truth.

CenterNet-style, one pass: a gate-centre heatmap plus 8 offset channels (the 4 inner corners relative to the
centre), so several gates in view are grouped without a separate association step. Output stride 2.
Inputs are 3xHxW RGB in [0,1]; corners are returned in pixels, order tl, tr, br, bl (inner opening).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

OFF_SCALE = 32.0  # offsets are regressed in units of 32 px


def _cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class GateNet(nn.Module):
    def __init__(self, w: int = 24, refine: bool = False):
        super().__init__()
        self.refine = refine
        self.e1 = nn.Sequential(_cbr(3, w, 2), _cbr(w, w))            # 1/2
        self.e2 = nn.Sequential(_cbr(w, 2 * w, 2), _cbr(2 * w, 2 * w))  # 1/4
        self.e3 = nn.Sequential(_cbr(2 * w, 4 * w, 2), _cbr(4 * w, 4 * w))  # 1/8
        self.e4 = nn.Sequential(_cbr(4 * w, 6 * w, 2), _cbr(6 * w, 6 * w))  # 1/16
        self.d3 = _cbr(6 * w + 4 * w, 4 * w)
        self.d2 = _cbr(4 * w + 2 * w, 2 * w)
        self.d1 = _cbr(2 * w + w, 2 * w)
        self.head = nn.Conv2d(2 * w, 1 + 8 + (12 if refine else 0), 1)  # centre, 8 corner offsets, [4 corner heatmaps, 8 sub-cell offsets]
        nn.init.constant_(self.head.bias[:1], -2.5)
        if refine:
            nn.init.constant_(self.head.bias[9:13], -2.5)

    def forward(self, x):
        x = x - 0.5
        a = self.e1(x); b = self.e2(a); c = self.e3(b); d = self.e4(c)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        y = self.d3(torch.cat((up(d, c), c), 1))
        y = self.d2(torch.cat((up(y, b), b), 1))
        y = self.d1(torch.cat((up(y, a), a), 1))
        self._y = y  # decoder feature (2w channels, stride 2): shared with the opponent head (droneracing/oppnet.py)
        o = self.head(y)
        if self.refine:
            return o[:, :1], o[:, 1:9] * OFF_SCALE, o[:, 9:13], o[:, 13:21]
        return o[:, :1], o[:, 1:9] * OFF_SCALE  # centre logits [B,1,H/2,W/2], offsets [B,8,...] in px

    @torch.no_grad()
    def detect(self, x, thr=0.3, k=6):
        """Image batch -> list of [n,9] detections (score + 4 corners in px), corners refined by the corner heatmaps if available."""
        out = self(x)
        if self.refine:
            return decode_refined(*out, thr=thr, k=k)
        return decode(out[0], out[1], thr=thr, k=k)


def make_targets(corners_px, vis, slot_valid, H, W, stride=2):
    """corners_px [B,S,4,2] (pixels), vis [B,S,4], slot_valid [B,S] -> heat [B,1,h,w], off [B,8,h,w], off_mask [B,8,h,w], pos [B,1,h,w]."""
    B, S = corners_px.shape[:2]
    h, w = H // stride, W // stride
    dev = corners_px.device
    ctr = corners_px.mean(2)  # [B,S,2]
    inside = (ctr[..., 0] >= 0) & (ctr[..., 0] < W) & (ctr[..., 1] >= 0) & (ctr[..., 1] < H)
    sane = (corners_px.abs() < 4 * max(H, W)).all(-1).all(-1)
    pos_s = slot_valid & inside & (vis.sum(-1) >= 2) & sane  # [B,S]
    ext = (corners_px.amax(2) - corners_px.amin(2)).amax(-1)  # px extent
    sigma = (ext / (12.0 * stride)).clamp(1.0, 6.0)
    ys = torch.arange(h, device=dev).view(1, 1, h, 1).float() + 0.5
    xs = torch.arange(w, device=dev).view(1, 1, 1, w).float() + 0.5
    cx = ((ctr[..., 0] / stride).floor() + 0.5)[..., None, None]  # snap the peak to a cell centre -> exactly 1.0
    cy = ((ctr[..., 1] / stride).floor() + 0.5)[..., None, None]
    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma[..., None, None] ** 2)) * pos_s[..., None, None]
    heat = g.amax(1, keepdim=True)
    off = torch.zeros(B, 8, h, w, device=dev)
    mask = torch.zeros(B, 8, h, w, device=dev)
    ci = (ctr[..., 0] / stride).long().clamp(0, w - 1)
    cj = (ctr[..., 1] / stride).long().clamp(0, h - 1)
    rel = (corners_px - ctr[:, :, None, :]).reshape(B, S, 8)  # (dx,dy) x4
    cin = ((corners_px[..., 0] > -0.5 * W) & (corners_px[..., 0] < 1.5 * W) & (corners_px[..., 1] > -0.5 * H) & (corners_px[..., 1] < 1.5 * H))
    cmask = cin.repeat_interleave(2, -1).float()  # [B,S,8]
    bi = torch.arange(B, device=dev)
    for s in range(S):
        ok = pos_s[:, s]
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                jj, ii = (cj[:, s] + dj).clamp(0, h - 1), (ci[:, s] + di).clamp(0, w - 1)
                # write only for positives (later slots overwrite earlier ones on collision, acceptable)
                off[bi[ok], :, jj[ok], ii[ok]] = rel[ok, s]
                mask[bi[ok], :, jj[ok], ii[ok]] = cmask[ok, s]
    return heat, off, mask, pos_s


def make_corner_targets(corners_px, vis, slot_valid, H, W, stride=2, sigma=1.5):
    """Corner heatmaps [B,4,h,w] (peak 1.0 at the cell holding the corner) and sub-cell offsets [B,8,h,w] + mask [B,8,h,w]."""
    B, S = corners_px.shape[:2]
    h, w = H // stride, W // stride
    dev = corners_px.device
    ctr = corners_px.mean(2)
    inside_c = (ctr[..., 0] >= 0) & (ctr[..., 0] < W) & (ctr[..., 1] >= 0) & (ctr[..., 1] < H)
    sane = (corners_px.abs() < 4 * max(H, W)).all(-1).all(-1)
    ok_s = slot_valid & inside_c & (vis.sum(-1) >= 2) & sane  # same positives as the centre heatmap
    cin = (corners_px[..., 0] >= 0) & (corners_px[..., 0] < W) & (corners_px[..., 1] >= 0) & (corners_px[..., 1] < H)
    pos = ok_s[..., None] & cin & vis  # [B,S,4]
    ys = torch.arange(h, device=dev).view(1, 1, 1, h, 1).float() + 0.5
    xs = torch.arange(w, device=dev).view(1, 1, 1, 1, w).float() + 0.5
    cxs = ((corners_px[..., 0] / stride).floor() + 0.5)[..., None, None]  # [B,S,4,1,1]
    cys = ((corners_px[..., 1] / stride).floor() + 0.5)[..., None, None]
    g = torch.exp(-((xs - cxs) ** 2 + (ys - cys) ** 2) / (2 * sigma**2)) * pos[..., None, None]
    heat = g.amax(1)  # [B,4,h,w]
    off = torch.zeros(B, 8, h, w, device=dev)
    mask = torch.zeros(B, 8, h, w, device=dev)
    ci = (corners_px[..., 0] / stride).long().clamp(0, w - 1)
    cj = (corners_px[..., 1] / stride).long().clamp(0, h - 1)
    frac = corners_px / stride - torch.stack((ci, cj), -1).float() - 0.5  # [B,S,4,2] in [-0.5, 0.5]
    bi = torch.arange(B, device=dev)
    for s in range(S):
        for c in range(4):
            ok = pos[:, s, c]
            if ok.any():
                off[bi[ok], 2 * c, cj[ok, s, c], ci[ok, s, c]] = frac[ok, s, c, 0]
                off[bi[ok], 2 * c + 1, cj[ok, s, c], ci[ok, s, c]] = frac[ok, s, c, 1]
                mask[bi[ok], 2 * c, cj[ok, s, c], ci[ok, s, c]] = 1.0
                mask[bi[ok], 2 * c + 1, cj[ok, s, c], ci[ok, s, c]] = 1.0
    return heat, off, mask


def focal_loss(logits, heat):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1.0).float()
    neg_w = (1 - heat) ** 4
    lp = -((1 - p) ** 2) * torch.log(p) * pos
    ln = -(p**2) * torch.log(1 - p) * neg_w * (1 - pos)
    n = pos.sum().clamp_min(1.0)
    return (lp.sum() + ln.sum()) / n


@torch.no_grad()
def decode(logits, off, thr=0.3, k=6, stride=2):
    """-> list (per image) of tensors [n,9]: score, 4 corners (x,y) in pixels."""
    heat = torch.sigmoid(logits)
    keep = (F.max_pool2d(heat, 3, 1, 1) == heat).float() * heat
    B, _, h, w = keep.shape
    sc, idx = keep.view(B, -1).topk(k, dim=1)
    out = []
    for b in range(B):
        m = sc[b] > thr
        j, i = idx[b][m] // w, idx[b][m] % w
        cx, cy = (i.float() + 0.5) * stride, (j.float() + 0.5) * stride
        o = off[b][:, j, i].T.reshape(-1, 4, 2)
        pts = torch.stack((cx, cy), -1)[:, None, :] + o
        out.append(torch.cat((sc[b][m][:, None], pts.reshape(-1, 8)), -1))
    return out


@torch.no_grad()
def decode_refined(logits, off, ch_logits, ch_off, thr=0.3, k=6, stride=2, cthr=0.25):
    """Centre detections with their regressed corners, each corner snapped to the strongest corner-heatmap peak
    near it (window ~ 18 % of the gate extent, 5..24 px) and refined with the sub-cell offset."""
    base = decode(logits, off, thr=thr, k=k, stride=stride)
    ch = torch.sigmoid(ch_logits)
    B, _, h, w = ch.shape
    ys = (torch.arange(h, device=ch.device).float() + 0.5) * stride
    xs = (torch.arange(w, device=ch.device).float() + 0.5) * stride
    out = []
    for b, d in enumerate(base):
        if len(d) == 0:
            out.append(d)
            continue
        pts = d[:, 1:].view(-1, 4, 2).clone()
        ext = (pts.amax(1) - pts.amin(1)).amax(-1)
        r = (0.18 * ext).clamp(5.0, 24.0)
        for c in range(4):
            hc = ch[b, c]
            for n in range(len(d)):
                px, py = pts[n, c]
                win = ((xs[None, :] - px) ** 2 + (ys[:, None] - py) ** 2) <= r[n] ** 2
                if not win.any():
                    continue
                sc = torch.where(win, hc, torch.zeros_like(hc))
                v, flat = sc.view(-1).max(0)
                if v > cthr:
                    j, i = flat // w, flat % w
                    pts[n, c, 0] = (i.float() + 0.5 + ch_off[b, 2 * c, j, i]) * stride
                    pts[n, c, 1] = (j.float() + 0.5 + ch_off[b, 2 * c + 1, j, i]) * stride
        out.append(torch.cat((d[:, :1], pts.reshape(-1, 8)), -1))
    return out
