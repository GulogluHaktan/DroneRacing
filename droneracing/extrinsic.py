"""Camera mount (extrinsic rotation) error: correction of the detections and offline calibration from a flight log.

A rotation E of the camera against its nominal mount (true camera = nominal camera * E) moves every image point by the
homography K E^T K^-1, independent of the depth. The ray of a measured pixel in the nominal camera frame is d_nom = E d_meas.
So a known E is removed by correcting the detected pixels before the filter; the filter and its camera model stay unchanged.

Calibration (MonoRace idea, arXiv 2601.15222: refine the extrinsic angles from onboard data after a flight): replay the filter on
the logged IMU and detections with a candidate E, and score the agreement between the corrected detections and the gates that the
filter predicts at the capture time. The candidate with the smallest robust corner residual wins. No ground truth is needed.
"""
from __future__ import annotations

import math

import torch

from .estimator import DET_DIM, K_DET, features_from_points


def rot_xyz(deg: torch.Tensor) -> torch.Tensor:
    """[...,3] angles (deg) about the camera x, y, z axes -> rotation [...,3,3] (R = Rz Ry Rx)."""
    a = torch.deg2rad(deg)
    cx, cy, cz = torch.cos(a[..., 0]), torch.cos(a[..., 1]), torch.cos(a[..., 2])
    sx, sy, sz = torch.sin(a[..., 0]), torch.sin(a[..., 1]), torch.sin(a[..., 2])
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack((o, z, z, z, cx, -sx, z, sx, cx), -1).reshape(*cx.shape, 3, 3)
    Ry = torch.stack((cy, z, sy, z, o, z, -sy, z, cy), -1).reshape(*cx.shape, 3, 3)
    Rz = torch.stack((cz, -sz, z, sz, cz, z, z, z, o), -1).reshape(*cx.shape, 3, 3)
    return Rz @ Ry @ Rx


def correct_px(px: torch.Tensor, E: torch.Tensor, f: float, cx: float, cy: float) -> torch.Tensor:
    """px [B,...,2] measured pixels, E [B,3,3] (true = nominal * E) -> pixels of the nominal camera."""
    B = px.shape[0]
    shp = px.shape
    p = px.reshape(B, -1, 2)
    d = torch.stack(((p[..., 0] - cx) / f, (p[..., 1] - cy) / f, torch.ones_like(p[..., 0])), -1)  # [B,M,3]
    dn = d @ E.transpose(1, 2)
    z = dn[..., 2].clamp_min(1e-3)
    return torch.stack((f * dn[..., 0] / z + cx, f * dn[..., 1] / z + cy), -1).reshape(shp)


def correct_feat(feat: torch.Tensor, E: torch.Tensor, H: int, W: int, hfov_deg: float, cam_tilt: float) -> torch.Tensor:
    """Detection features [B, K*DET_DIM] -> same features rebuilt from the corrected corners."""
    B = feat.shape[0]
    fd = feat.reshape(B, K_DET, DET_DIM)
    det = fd[:, :, 0] > 0
    wh = torch.tensor([W, H], device=feat.device, dtype=feat.dtype)
    pts = (fd[:, :, 8:16].reshape(B, K_DET, 4, 2) + 1.0) * 0.5 * wh
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    pts = correct_px(pts, E, f, W / 2, H / 2)
    out = features_from_points(pts, fd[:, :, 0], det, H, W, hfov_deg=hfov_deg, cam_tilt=cam_tilt)
    return torch.where(det.repeat_interleave(DET_DIM, 1), out, feat)


def distort_feat(feat: torch.Tensor, E: torch.Tensor, H: int, W: int, hfov_deg: float, cam_tilt: float) -> torch.Tensor:
    """Inverse of correct_feat: what a camera with mount error E measures (for tests on logs without a mount error)."""
    return correct_feat(feat, E.transpose(1, 2), H, W, hfov_deg, cam_tilt)
