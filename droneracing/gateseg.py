"""Gate-frame segmentation + edge-line corner extraction (MonoRace GateNet + QuAdGate style).

The network predicts a binary mask of the gate FRAME (outer square minus inner opening). Corners are not regressed: for each
connected frame region, straight lines are fitted to the inner and outer edges and intersected, giving 4 inner + 4 outer
corners with sub-pixel accuracy. A line needs only a visible part of its edge, so corners outside the image (close, partial
gates) are extrapolated. Mask labels come for free from the 4 labelled inner corners: the gate plane -> image homography maps
the outer square (half side `outer_half`) into the image.

Outputs use the GateNet detection format ([score, 4 inner corners (x, y) px] rows) so det_features / PnP / EKF stay unchanged.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gatenet import _cbr

INNER_HALF = 0.75  # A2RL gate: 1.5 m opening
OUTER_HALF = 1.35  # 0.6 m frame -> 2.7 m outer side


class GateSeg(nn.Module):
    """GateNet-sized U-Net, one output channel: frame-mask logits at full input resolution."""

    def __init__(self, w: int = 24):
        super().__init__()
        self.e1 = nn.Sequential(_cbr(3, w, 2), _cbr(w, w))
        self.e2 = nn.Sequential(_cbr(w, 2 * w, 2), _cbr(2 * w, 2 * w))
        self.e3 = nn.Sequential(_cbr(2 * w, 4 * w, 2), _cbr(4 * w, 4 * w))
        self.e4 = nn.Sequential(_cbr(4 * w, 6 * w, 2), _cbr(6 * w, 6 * w))
        self.d3 = _cbr(6 * w + 4 * w, 4 * w)
        self.d2 = _cbr(4 * w + 2 * w, 2 * w)
        self.d1 = _cbr(2 * w + w, 2 * w)
        self.d0 = _cbr(2 * w + 3, w)
        self.head = nn.Conv2d(w, 1, 1)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, x):
        x0 = x - 0.5
        a = self.e1(x0); b = self.e2(a); c = self.e3(b); d = self.e4(c)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        y = self.d3(torch.cat((up(d, c), c), 1))
        y = self.d2(torch.cat((up(y, b), b), 1))
        y = self.d1(torch.cat((up(y, a), a), 1))
        y = self.d0(torch.cat((up(y, x0), x0), 1))
        return self.head(y)  # [B,1,H,W]


# ---- labels -------------------------------------------------------------------------------------------------------------
def _homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """DLT: 4 point pairs src [M,4,2] -> dst [M,4,2], returns H [M,3,3] (dst ~ H src)."""
    M = src.shape[0]
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    z, o = torch.zeros_like(x), torch.ones_like(x)
    r1 = torch.stack((x, y, o, z, z, z, -u * x, -u * y, -u), -1)
    r2 = torch.stack((z, z, z, x, y, o, -v * x, -v * y, -v), -1)
    A = torch.cat((r1, r2), 1)  # [M,8,9]
    _, _, Vh = torch.linalg.svd(A)
    return Vh[:, -1].reshape(M, 3, 3)


def _inside_quad(px: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """px [P,2] grid points, q [M,4,2] convex quads (any winding) -> [M,P] bool."""
    e = torch.roll(q, -1, 1) - q  # [M,4,2]
    r = px[None, None] - q[:, :, None]  # [M,4,P,2]
    cr = e[:, :, None, 0] * r[..., 1] - e[:, :, None, 1] * r[..., 0]  # [M,4,P]
    return (cr >= 0).all(1) | (cr <= 0).all(1)


GATE_PLANE = torch.tensor([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])  # corner order of the labels (TL, TR, BR, BL)


def frame_masks(corners_px: torch.Tensor, slot_valid: torch.Tensor, H: int, W: int, stride: int = 1) -> torch.Tensor:
    """Frame-mask labels [B,1,H/stride,W/stride] (float 0/1) from inner corners [B,G,4,2] px of valid gate slots [B,G].

    Gates partly behind the camera or with degenerate projections are skipped (their pixels stay 0)."""
    B, G = slot_valid.shape
    dev = corners_px.device
    h, w = H // stride, W // stride
    ys, xs = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    grid = (torch.stack((xs, ys), -1).reshape(-1, 2).float() + 0.5) * stride
    out = torch.zeros(B, h * w, device=dev)
    c = corners_px.float()
    ok = slot_valid & torch.isfinite(c).all(-1).all(-1) & (c.abs() < 20 * max(H, W)).all(-1).all(-1)
    bi, gi = torch.nonzero(ok, as_tuple=True)
    if len(bi) == 0:
        return out.reshape(B, 1, h, w)
    inner = c[bi, gi]  # [M,4,2]
    Hm = _homography((GATE_PLANE.to(dev) * INNER_HALF)[None].expand(len(bi), 4, 2), inner)
    po = torch.cat(((GATE_PLANE.to(dev) * OUTER_HALF)[None].expand(len(bi), 4, 2), torch.ones(len(bi), 4, 1, device=dev)), -1)
    pr = (Hm[:, None] @ po[..., None])[..., 0]
    good = ((pr[..., 2] > 1e-6).all(-1) | (pr[..., 2] < -1e-6).all(-1))  # outer corners on one side of the camera
    outer = pr[..., :2] / pr[..., 2:3]
    m = _inside_quad(grid, outer) & ~_inside_quad(grid, inner)  # [M,P]
    m = m & good[:, None]
    out.index_put_((bi,), m.float(), accumulate=True)
    return out.clamp(max=1.0).reshape(B, 1, h, w)


# ---- corner extraction --------------------------------------------------------------------------------------------------
def _fit_line(pts: np.ndarray):
    """Least-squares line through pts [N,2] -> (point, unit direction) or None."""
    if len(pts) < 4:
        return None
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(l1, l2):
    (p1, d1), (p2, d2) = l1, l2
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def _quad_lines(contour: np.ndarray, W: int, H: int, border: int = 2):
    """Contour [N,2] of one edge loop -> 4 fitted side lines in loop order, using only points away from the image border."""
    peri = cv2.arcLength(contour.reshape(-1, 1, 2).astype(np.float32), True)
    hull = cv2.convexHull(contour.reshape(-1, 1, 2).astype(np.float32))
    approx = None
    for eps in (0.02, 0.04, 0.06, 0.08, 0.12):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4:
            approx = a.reshape(4, 2)
            break
    if approx is None:
        return None
    pts = contour.reshape(-1, 2).astype(np.float64)
    keep = (pts[:, 0] > border) & (pts[:, 0] < W - 1 - border) & (pts[:, 1] > border) & (pts[:, 1] < H - 1 - border)
    # assign each contour point to the nearest side of the approximating quad
    a = approx.astype(np.float64)
    b = np.roll(a, -1, 0)
    d = b - a
    t = np.clip(((pts[:, None] - a[None]) * d[None]).sum(-1) / (d * d).sum(-1)[None], 0, 1)
    proj = a[None] + t[..., None] * d[None]
    side = np.linalg.norm(pts[:, None] - proj, axis=-1).argmin(1)
    lines = []
    for s in range(4):
        sel = pts[(side == s) & keep]
        ln = _fit_line(sel) if len(sel) >= 6 else None
        if ln is None:  # side not visible: use the approximating quad side (border-clipped)
            ln = (a[s], d[s] / (np.linalg.norm(d[s]) + 1e-9))
        lines.append(ln)
    return lines


def _order_tl(q: np.ndarray) -> np.ndarray:
    """4 corners -> (TL, TR, BR, BL) by angle around the centroid (image y down), start at the top-left one."""
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]  # clockwise in image coordinates, starting near -pi (left)
    s = q.sum(1)
    k = int(np.argmin(s))
    return np.roll(q, -k, 0)


def extract_gates(mask: np.ndarray, prob: np.ndarray | None = None, min_area: int = 30, k: int = 6):
    """Binary frame mask [H,W] -> list of (score, inner [4,2], outer [4,2] or None), largest gates first."""
    H, W = mask.shape
    m8 = mask.astype(np.uint8)
    contours, hier = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:  # holes are handled with their parent
            continue
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        outer_lines = _quad_lines(cnt, W, H)
        hole = hier[i][2]
        inner_lines = None
        best_hole, best_area = None, 0.0
        while hole != -1:
            ha = cv2.contourArea(contours[hole])
            if ha > best_area:
                best_hole, best_area = hole, ha
            hole = hier[hole][0]
        if best_hole is not None and best_area > 0.05 * area:
            inner_lines = _quad_lines(contours[best_hole], W, H)
        outer = inner = None
        if outer_lines is not None:
            pts = [_intersect(outer_lines[s], outer_lines[(s + 1) % 4]) for s in range(4)]
            if all(p is not None for p in pts):
                outer = _order_tl(np.array([pts[3], pts[0], pts[1], pts[2]]))
        if inner_lines is not None:
            pts = [_intersect(inner_lines[s], inner_lines[(s + 1) % 4]) for s in range(4)]
            if all(p is not None for p in pts):
                inner = _order_tl(np.array([pts[3], pts[0], pts[1], pts[2]]))
        if inner is None and outer is not None:  # opening not visible as a closed hole: shrink the outer quad in the gate plane
            Hm = cv2.getPerspectiveTransform((GATE_PLANE.numpy() * OUTER_HALF).astype(np.float32), outer.astype(np.float32))
            inner = cv2.perspectiveTransform((GATE_PLANE.numpy() * INNER_HALF).astype(np.float32)[None], Hm)[0].astype(np.float64)
        if inner is None or not np.isfinite(inner).all() or np.abs(inner).max() > 20 * max(H, W):
            continue
        sc = float(prob[m8 > 0].mean()) if prob is not None else 1.0
        out.append((sc, inner, outer, area))
    out.sort(key=lambda r: -r[3])
    return [(s, i, o) for s, i, o, _ in out[:k]]


@torch.no_grad()
def reorder_by_roll(corners: np.ndarray, roll: float, cx: float, cy: float) -> np.ndarray:
    """Corners [4,2] in image order -> gate-frame order TL, TR, BR, BL for a camera rolled by ``roll`` [rad] about the optical axis
    (angle from the image up direction to the world up direction, clockwise; MonoRace derotates the mask with the filter attitude)."""
    c, s = math.cos(-roll), math.sin(-roll)
    d = corners - np.array([cx, cy])
    r = np.stack((c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]), -1)  # derotated: world up = image up
    top = np.argsort(r[:, 1])[:2]
    bot = np.argsort(r[:, 1])[2:]
    tl, tr = top[np.argsort(r[top, 0])]
    bl, br = bot[np.argsort(r[bot, 0])]
    return corners[[tl, tr, br, bl]]


def detect(net: GateSeg, img: torch.Tensor, thr: float = 0.5, k: int = 6, roll: torch.Tensor | None = None):
    """Image batch [B,3,H,W] -> list of [n,9] tensors (score + 4 inner corners px, TL TR BR BL) like gatenet.decode."""
    logit = net(img)
    prob = torch.sigmoid(logit)[:, 0].cpu().numpy()
    res = []
    for b in range(img.shape[0]):
        g = extract_gates(prob[b] > thr, prob[b], k=k)
        H_, W_ = img.shape[-2:]
        rows = [torch.tensor([s] + list((inner if roll is None else reorder_by_roll(np.asarray(inner, dtype=np.float64), float(roll[b]), W_ / 2, H_ / 2)).reshape(-1)), dtype=torch.float32) for s, inner, _ in g]
        res.append(torch.stack(rows) if rows else torch.zeros(0, 9))
    return res
