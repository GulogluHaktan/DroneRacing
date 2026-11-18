"""Rolling-shutter compensation of corner measurements with the gyro (row time per corner).

A rolling-shutter camera captures row r at t_r = (r / H - 0.5) * T (T = readout time, t = 0 at the frame centre). A camera that
rotates with omega_c (camera frame: x right, y down, z forward) shows a corner of row r shifted by f * (-omega_y, omega_x) * t_r
(same model as isaac/cam_artifacts.rolling_shutter). The detector finds the corner where it appears in the image; the filter uses
one capture time per frame. Moving each corner back by the shift of its own row puts all corners at the frame-centre time.
The translation part (forward motion, v / Z) is not corrected here.
"""
from __future__ import annotations

import math

import torch

from .estimator import DET_DIM, K_DET, features_from_points


def rs_correct_px(px: torch.Tensor, omega_c: torch.Tensor, f: float, H: int, readout_s: float) -> torch.Tensor:
    """px [B,...,2] measured pixels, omega_c [B,3] -> pixels at the frame-centre time.
    The image row where a corner appears (the measured row) was read at t = (v / H - 0.5) * T, so the exact inverse is one step."""
    shp = [px.shape[0]] + [1] * (px.dim() - 2)
    wx, wy = omega_c[:, 0].view(*shp), omega_c[:, 1].view(*shp)
    u, v = px[..., 0], px[..., 1]
    t = ((v + 0.5) / H - 0.5) * readout_s
    return torch.stack((u - f * -wy * t, v - f * wx * t), -1)


def rs_correct_feat(feat: torch.Tensor, omega_c: torch.Tensor, H: int, W: int, hfov_deg: float, cam_tilt: float, readout_s: float) -> torch.Tensor:
    """Detection features [B, K*DET_DIM] -> rebuilt from the corrected corners."""
    B = feat.shape[0]
    fd = feat.reshape(B, K_DET, DET_DIM)
    det = fd[:, :, 0] > 0
    wh = torch.tensor([W, H], device=feat.device, dtype=feat.dtype)
    pts = (fd[:, :, 8:16].reshape(B, K_DET, 4, 2) + 1.0) * 0.5 * wh
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    pts = rs_correct_px(pts, omega_c, f, H, readout_s)
    out = features_from_points(pts, fd[:, :, 0], det, H, W, hfov_deg=hfov_deg, cam_tilt=cam_tilt)
    return torch.where(det.repeat_interleave(DET_DIM, 1), out, feat)
