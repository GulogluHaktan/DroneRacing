"""Adaptive crop around the predicted gate (MonoRace: 384x384 window of the 820x616 frame, +36 % corners, +25 % PnP frames).

The detector sees a fixed ``out`` x ``out`` window. Its side in full-image pixels is the predicted gate extent times a margin, but
never smaller than ``out`` (so close gates are downscaled to fit and far gates are seen at full resolution). Without a usable
prediction the whole frame is letterboxed into the window. Detections are mapped back to full-image pixels, so the EKF works
with the full-frame intrinsics.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def crop_windows(pred_uv: torch.Tensor, ok: torch.Tensor, H: int, W: int, out: int = 384, margin: float = 1.6) -> torch.Tensor:
    """pred_uv [n,4,2] predicted gate corners (full px), ok [n] bool -> windows [n,3] (x0, y0, side) in full px."""
    lo, hi = pred_uv.amin(1), pred_uv.amax(1)
    ctr = (lo + hi) / 2
    side = ((hi - lo).amax(-1) * margin).clamp(min=out, max=max(H, W))
    full = torch.full_like(side, float(max(H, W)))
    side = torch.where(ok, side, full)
    ctr = torch.where(ok[:, None], ctr, torch.tensor([W / 2, H / 2], device=pred_uv.device).expand_as(ctr))
    # keep the window inside the frame where possible (a window larger than the frame is centred on it)
    x0 = torch.where(side >= W, (W - side) / 2, torch.minimum((ctr[:, 0] - side / 2).clamp_min(0), W - side))
    y0 = torch.where(side >= H, (H - side) / 2, torch.minimum((ctr[:, 1] - side / 2).clamp_min(0), H - side))
    return torch.stack((x0, y0, side), -1)


def extract(img: torch.Tensor, win: torch.Tensor, out: int = 384) -> torch.Tensor:
    """img [n,C,H,W], win [n,3] -> [n,C,out,out] (bilinear, zero outside the frame)."""
    n, _, H, W = img.shape
    t = (torch.arange(out, device=img.device, dtype=img.dtype) + 0.5) / out  # [out]
    xs = win[:, 0:1] + t[None] * win[:, 2:3]  # [n,out] full-image px
    ys = win[:, 1:2] + t[None] * win[:, 2:3]
    gx = xs / W * 2 - 1
    gy = ys / H * 2 - 1
    grid = torch.stack((gx[:, None, :].expand(n, out, out), gy[:, :, None].expand(n, out, out)), -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def to_full(uv: torch.Tensor, win_b: torch.Tensor, out: int = 384) -> torch.Tensor:
    """Pixel coords [...,2] in the window of one sample (win_b [3]) -> full-image px."""
    return win_b[:2] + uv * (win_b[2] / out)


def to_crop(uv: torch.Tensor, win: torch.Tensor, out: int = 384) -> torch.Tensor:
    """Full-image px [n,...,2] -> window px, win [n,3]."""
    shp = [win.shape[0]] + [1] * (uv.dim() - 2)
    return (uv - win[:, :2].view(*shp, 2)) * (out / win[:, 2]).view(*shp, 1)


def dets_to_full(dets: list, win: torch.Tensor, out: int = 384) -> list:
    """GateNet decode output (list of [k,9]: score + 4 corners px of the window) -> same in full-image px."""
    res = []
    for b, d in enumerate(dets):
        if len(d) == 0:
            res.append(d)
            continue
        d = d.clone()
        d[:, 1:] = to_full(d[:, 1:].reshape(-1, 4, 2), win[b], out).reshape(-1, 8)
        res.append(d)
    return res


def peaks_to_full(peaks: list, win: torch.Tensor, out: int = 384) -> list:
    """decode_corner_peaks output (list of [k,4]: type, score, u, v in window px) -> full-image px."""
    res = []
    for b, p in enumerate(peaks):
        if len(p) == 0:
            res.append(p)
            continue
        p = p.clone()
        p[:, 2:4] = to_full(p[:, 2:4], win[b], out)
        res.append(p)
    return res
