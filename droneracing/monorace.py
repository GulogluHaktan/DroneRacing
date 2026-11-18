"""MonoRace perception (Bahnam et al., arXiv 2601.15222) re-implemented for our camera line.

1. ``GateNetMR``: U-Net with channel factor f (f = 4), encoder-decoder with additive skips, five outputs at 1/16, 1/8, 1/4, 1/2 and full resolution
   (deep supervision: Dice + 2 BCE per output, weights 4, 2, 1, 1, 1 from the full resolution down), input = the adaptive crop around the predicted gates.
2. ``quadgate``: QuAdGate corner extraction. The mask is derotated with the filter attitude (vertical axis of the image = world up); line segments from the
   OpenCV LSD (scale 0.8, sigma_scale 0.8, quant 25, ang_th 30); each line is extended to 5/3 of its length; the intersections are corner candidates; each
   candidate gets a 4-value descriptor of the mask (5 px along the axes: TL, TR, BR, BL); the priors (corners of the next gates projected with the filter state:
   4 inner and 4 outer corners per gate) are matched to the candidates of the same descriptor within 100 px; a partial affine transform (translation,
   rotation, uniform scale; RANSAC, 5 px) rejects wrong matches; a translation above 150 px rejects the frame.
The result is a list of corner measurements with known identity (gate, corner type 0-7), used by ``GateEKF8``.
Corner types: 0-3 inner TL, TR, BR, BL (opening 1.5 m); 4-7 outer TL, TR, BR, BL (2.7 m).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INNER_HALF, OUTER_HALF = 0.75, 1.35
# (right sign, up sign, half size) of each corner type in the gate plane
TYPE_GEOM = ((-1, 1, INNER_HALF), (1, 1, INNER_HALF), (1, -1, INNER_HALF), (-1, -1, INNER_HALF),
             (-1, 1, OUTER_HALF), (1, 1, OUTER_HALF), (1, -1, OUTER_HALF), (-1, -1, OUTER_HALF))
# mask values around a corner at (TL, TR, BR, BL) offsets: frame = 1, empty = 0 (paper: top-right outer corner = [0,0,0,1])
DESC = ((1, 1, 0, 1), (1, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 1), (0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, 0))


# ---------------------------------------------------------------------------------------------------- network
def _dc(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _Up(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.ct = nn.ConvTranspose2d(cin, skip, 2, stride=2)
        self.bn = nn.BatchNorm2d(skip)
        self.dc = _dc(skip, cout)

    def forward(self, x, s):
        return self.dc(self.bn(self.ct(x)) + s)


class GateNetMR(nn.Module):
    """Input [B,3,S,S] (S divisible by 16) -> five logit maps [B,1,S/16 ..., S] (deepest first)."""

    def __init__(self, f: int = 4):
        super().__init__()
        c = [64 // f, 128 // f, 256 // f, 512 // f, 512 // f]
        self.inc, self.d1, self.d2, self.d3, self.d4 = _dc(3, c[0]), _dc(c[0], c[1]), _dc(c[1], c[2]), _dc(c[2], c[3]), _dc(c[3], c[4])
        self.up1, self.up2, self.up3, self.up4 = _Up(c[4], c[3], c[2]), _Up(c[2], c[2], c[1]), _Up(c[1], c[1], c[0]), _Up(c[0], c[0], c[0])
        self.o0, self.o1, self.o2, self.o3, self.o4 = (nn.Conv2d(ch, 1, 1) for ch in (c[4], c[2], c[1], c[0], c[0]))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(F.max_pool2d(x1, 2))
        x3 = self.d2(F.max_pool2d(x2, 2))
        x4 = self.d3(F.max_pool2d(x3, 2))
        x5 = self.d4(F.max_pool2d(x4, 2))
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        return [self.o0(x5), self.o1(u1), self.o2(u2), self.o3(u3), self.o4(u4)]


def deep_loss(outs, target):
    """target [B,1,S,S] in 0..1. L = 4 L0 + 2 L1 + L2 + L3 + L4 with L_i = Dice + 2 BCE at the resolution of output i (index 4 = full)."""
    w = (1.0, 1.0, 1.0, 2.0, 4.0)
    tot = 0.0
    for o, wi in zip(outs, w):
        t = F.interpolate(target, size=o.shape[-2:], mode="area")
        p = torch.sigmoid(o)
        dice = 1 - (2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
        bce = F.binary_cross_entropy_with_logits(o, t)
        tot = tot + wi * (dice.mean() + 2 * bce)
    return tot


@torch.no_grad()
def predict_masks(net: GateNetMR, crops: torch.Tensor) -> np.ndarray:
    """crops [B,3,S,S] 0..1 -> probability maps [B,S,S] (highest resolution output)."""
    return torch.sigmoid(net(crops)[-1])[:, 0].cpu().numpy()


# ---------------------------------------------------------------------------------------------------- QuAdGate
_LSD = None


def _lsd():
    global _LSD
    if _LSD is None:
        _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, 0.8, 0.8, 25.0, 30.0)
    return _LSD


def derot_matrix(roll: float, cx: float, cy: float) -> np.ndarray:
    """2x3 matrix that maps image points to the derotated frame (world up = image up): p' = R(-roll)(p - c) + c."""
    c, s = math.cos(-roll), math.sin(-roll)
    R = np.array([[c, -s], [s, c]])
    return np.concatenate((R, (np.array([cx, cy]) - R @ np.array([cx, cy]))[:, None]), 1)


def _apply(A: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ A[:, :2].T + A[:, 2]


def _candidates(mask_u8: np.ndarray, ext: float = 5.0 / 3.0, min_len: float = 6.0):
    """Corner candidates (x, y) = intersections of the extended LSD lines (angle between 40 and 140 deg)."""
    res = _lsd().detect(mask_u8)[0]
    if res is None or len(res) < 2:
        return np.zeros((0, 2))
    L = res[:, 0, :].astype(np.float64)  # x1 y1 x2 y2
    d = L[:, 2:] - L[:, :2]
    ln = np.linalg.norm(d, axis=1)
    keep = ln > min_len
    L, d, ln = L[keep], d[keep], ln[keep]
    if len(L) < 2:
        return np.zeros((0, 2))
    mid = (L[:, :2] + L[:, 2:]) / 2
    u = d / ln[:, None]
    p0, p1 = mid - u * ln[:, None] * ext / 2, mid + u * ln[:, None] * ext / 2
    out = []
    H, W = mask_u8.shape
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            cosang = abs(float(u[i] @ u[j]))
            if cosang > 0.77:  # nearly parallel
                continue
            # intersection of the infinite lines through the extended segments, kept only if it lies on both extended segments
            den = u[i, 0] * u[j, 1] - u[i, 1] * u[j, 0]
            if abs(den) < 1e-6:
                continue
            t = ((mid[j, 0] - mid[i, 0]) * u[j, 1] - (mid[j, 1] - mid[i, 1]) * u[j, 0]) / den
            s = ((mid[j, 0] - mid[i, 0]) * u[i, 1] - (mid[j, 1] - mid[i, 1]) * u[i, 0]) / den
            if abs(t) <= ln[i] * ext / 2 and abs(s) <= ln[j] * ext / 2:
                q = mid[i] + u[i] * t
                if -10 <= q[0] < W + 10 and -10 <= q[1] < H + 10:
                    out.append(q)
    return np.array(out) if out else np.zeros((0, 2))


def _descriptor(mask_b: np.ndarray, pts: np.ndarray, off: float) -> np.ndarray:
    H, W = mask_b.shape
    offs = np.array([[-off, -off], [off, -off], [off, off], [-off, off]])
    xy = np.rint(pts[:, None, :] + offs[None]).astype(int)
    ok = (xy[..., 0] >= 0) & (xy[..., 0] < W) & (xy[..., 1] >= 0) & (xy[..., 1] < H)
    v = np.zeros(ok.shape, dtype=np.int8)
    v[ok] = mask_b[xy[..., 1][ok], xy[..., 0][ok]]
    return np.where(ok, v, -1)  # -1: outside the image (unknown)


def quadgate(prob: np.ndarray, priors: np.ndarray, prior_gate: np.ndarray, prior_type: np.ndarray, roll: float,
             thr: float = 0.5, max_dist: float = 100.0, ransac_px: float = 5.0, max_shift: float = 150.0, min_match: int = 3):
    """One crop. prob [S,S]; priors [P,2] predicted corner pixels (crop coords) with the gate slot ``prior_gate`` and ``prior_type`` (0-7).
    Returns an array [M,5]: (gate slot, type, x, y, inlier flag 1) in crop coordinates, or an empty array."""
    S = prob.shape[0]
    if len(priors) == 0:
        return np.zeros((0, 5))
    A = derot_matrix(roll, S / 2, S / 2)
    mask = cv2.warpAffine((prob > thr).astype(np.uint8) * 255, A, (S, S), flags=cv2.INTER_NEAREST)
    cand = _candidates(mask)
    if len(cand) == 0:
        return np.zeros((0, 5))
    mb = (mask > 127).astype(np.uint8)
    pri = _apply(A, priors)
    size = max(float(np.ptp(pri[:, 0])), float(np.ptp(pri[:, 1])), 8.0)
    off = float(np.clip(0.4 * 0.222 * size, 2.0, 6.0))
    cd = _descriptor(mb, cand, off)
    src, dst, ids = [], [], []
    for k in range(len(pri)):
        want = np.array(DESC[int(prior_type[k])], dtype=np.int8)
        same = ((cd == want) | (cd < 0)).all(1) & ((cd == want).sum(1) >= 3)
        if not same.any():
            continue
        dd = np.linalg.norm(cand - pri[k], axis=1)
        dd = np.where(same, dd, 1e9)
        j = int(dd.argmin())
        if dd[j] < max_dist:
            src.append(pri[k]); dst.append(cand[j]); ids.append(k)
    if len(src) < min_match:
        return np.zeros((0, 5))
    M, inl = cv2.estimateAffinePartial2D(np.array(src, np.float32), np.array(dst, np.float32), method=cv2.RANSAC, ransacReprojThreshold=ransac_px)
    if M is None or inl is None or np.abs(M[:, 2]).max() > max_shift:
        return np.zeros((0, 5))
    inl = inl[:, 0].astype(bool)
    if inl.sum() < 2:
        return np.zeros((0, 5))
    Ainv = cv2.invertAffineTransform(A)
    out = []
    for q, k in zip(np.array(dst)[inl], np.array(ids)[inl]):
        x, y = _apply(Ainv, q[None])[0]
        out.append((prior_gate[k], prior_type[k], x, y, 1.0))
    return np.array(out)


def corner_priors(uv_cn: torch.Tensor, ok: torch.Tensor):
    """uv_cn [J,8,2] predicted pixel positions (crop coords) of the 8 corners of J gates; ok [J,8] bool -> priors, gate slots, types (numpy)."""
    j, t = torch.nonzero(ok, as_tuple=True)
    return uv_cn[j, t].cpu().numpy().astype(np.float64), j.cpu().numpy(), t.cpu().numpy()


def all_corners_from_inner(inner: torch.Tensor) -> torch.Tensor:
    """Inner corners [B,4,2] (TL, TR, BR, BL) -> the 8 corners [B,8,2] (inner then outer) through the gate-plane homography."""
    from .gateseg import GATE_PLANE, _homography

    dev = inner.device
    B = inner.shape[0]
    Hm = _homography((GATE_PLANE.to(dev) * INNER_HALF)[None].expand(B, 4, 2), inner.float())
    po = torch.cat(((GATE_PLANE.to(dev) * OUTER_HALF)[None].expand(B, 4, 2), torch.ones(B, 4, 1, device=dev)), -1)
    pr = (Hm[:, None] @ po[..., None])[..., 0]
    outer = pr[..., :2] / pr[..., 2:3].clamp_min(1e-6)
    return torch.cat((inner.float(), outer), 1)
