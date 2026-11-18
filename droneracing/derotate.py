"""Derotation of the detector input with the camera roll (MonoRace: the mask is derotated with the filter attitude).

After the derotation the world up direction points to the image top, so the gate-frame corner types (TL, TR, BR, BL) match their
image positions also when the drone is rolled or inverted. The detector then needs no roll-dependent corner labelling.
roll: angle of the world up direction in the image, measured from the image up direction (0 = level camera); the image is rotated
by -roll about its centre, and the detections are rotated back by +roll.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .quad import quat_to_rot


def cam_rb(cam_tilt: float, device) -> torch.Tensor:
    """Body -> camera axes (row-vector form used by the filters: X_cam = P_body @ Rb)."""
    s, c = math.sin(cam_tilt), math.cos(cam_tilt)
    return torch.tensor([[0.0, s, c], [-1.0, 0.0, 0.0], [0.0, -c, s]], device=device)


def roll_from_rot(R: torch.Tensor, cam_tilt: float) -> torch.Tensor:
    """R [B,3,3] body -> world rotation -> camera roll [B] (rad)."""
    up_cam = R[:, 2, :] @ cam_rb(cam_tilt, R.device)
    return torch.atan2(up_cam[:, 0], -up_cam[:, 1])


def roll_from_quat(q: torch.Tensor, cam_tilt: float) -> torch.Tensor:
    return roll_from_rot(quat_to_rot(q), cam_tilt)


def rotate_points(p: torch.Tensor, ang: torch.Tensor, cx: float, cy: float) -> torch.Tensor:
    """p [B,...,2] pixels -> rotated by ang [B] (rad, image coordinates x right, y down) about (cx, cy)."""
    shp = [p.shape[0]] + [1] * (p.dim() - 2)
    c, s = torch.cos(ang).view(*shp), torch.sin(ang).view(*shp)
    x, y = p[..., 0] - cx, p[..., 1] - cy
    return torch.stack((c * x - s * y + cx, s * x + c * y + cy), -1)


def derotate_img(img: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """img [B,C,H,W] -> image rotated by -roll about its centre (output pixel q samples the input at Rot(roll)(q - c) + c)."""
    B, C, H, W = img.shape
    dev, dt = img.device, img.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt) + 0.5, torch.arange(W, device=dev, dtype=dt) + 0.5, indexing="ij")
    q = torch.stack((xs, ys), -1)[None].expand(B, H, W, 2)
    src = rotate_points(q.reshape(B, -1, 2), roll.to(dt), W / 2, H / 2).reshape(B, H, W, 2)
    grid = torch.stack((src[..., 0] / W * 2 - 1, src[..., 1] / H * 2 - 1), -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def derotate_labels(px: torch.Tensor, roll: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Corner labels [B,...,2] in the original image -> positions in the derotated image."""
    return rotate_points(px, -roll, W / 2, H / 2)


def rerotate_points(px: torch.Tensor, roll: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Detections in the derotated image -> original image."""
    return rotate_points(px, roll, W / 2, H / 2)
