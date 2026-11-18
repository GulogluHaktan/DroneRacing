"""Camera artefacts of the real A2RL camera (IMX219 rolling shutter, radio interference) applied to rendered frames.

Used by ``eval_vision.py`` (``--rs-ms``, ``--corrupt``) to measure how much each artefact costs the camera line. Geometry of the
rolling shutter: a world-fixed direction seen by a camera rotating with ``omega_c`` (camera frame, x right, y down, z forward)
moves in the image by (du, dv) = f * (-omega_y, omega_x) per second near the image centre, so row r (captured at
t_r = (r / H - 0.5) * T) shows the scene shifted by that rate times t_r.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rolling_shutter(img: torch.Tensor, omega_c: torch.Tensor, f: float, readout_s: float) -> torch.Tensor:
    """img [B,C,H,W] in 0..1, omega_c [B,3] camera-frame angular rate [rad/s], f focal length [px], readout_s full-frame readout time."""
    B, _, H, W = img.shape
    dev = img.device
    t_r = ((torch.arange(H, device=dev) + 0.5) / H - 0.5) * readout_s  # [H]
    du = (f * -omega_c[:, 1])[:, None] * t_r[None]  # [B,H] pixel shift of row r
    dv = (f * omega_c[:, 0])[:, None] * t_r[None]
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=img.dtype), torch.arange(W, device=dev, dtype=img.dtype), indexing="ij")
    sx = xs[None] - du[:, :, None]
    sy = ys[None] - dv[:, :, None]
    grid = torch.stack(((sx + 0.5) / W * 2 - 1, (sy + 0.5) / H * 2 - 1), -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


def corrupt(img: torch.Tensor, p_frame: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """Radio-link interference as seen by MonoRace (8-75 % of frames): with probability p_frame a horizontal band covering 10-60 %
    of the rows is replaced by noise or a smeared copy of one row."""
    if p_frame <= 0:
        return img
    B, C, H, W = img.shape
    dev = img.device
    hit = torch.rand(B, device=dev, generator=gen) < p_frame
    if not hit.any():
        return img
    out = img.clone()
    for b in torch.nonzero(hit).flatten().tolist():
        h = int(H * (0.1 + 0.5 * float(torch.rand(1, device=dev, generator=gen))))
        y0 = int((H - h) * float(torch.rand(1, device=dev, generator=gen)))
        if float(torch.rand(1, device=dev, generator=gen)) < 0.5:
            out[b, :, y0 : y0 + h] = torch.rand(C, h, W, device=dev, generator=gen)
        else:
            out[b, :, y0 : y0 + h] = img[b, :, y0 : y0 + 1].expand(C, h, W)
    return out


def random_rotation(n: int, max_deg: float, device, gen: torch.Generator | None = None) -> torch.Tensor:
    """[n,3,3] rotations with each Euler angle uniform in +-max_deg (camera mount / extrinsic error)."""
    a = (torch.rand(n, 3, device=device, generator=gen) * 2 - 1) * max_deg * torch.pi / 180
    cx, sx, cy, sy, cz, sz = a[:, 0].cos(), a[:, 0].sin(), a[:, 1].cos(), a[:, 1].sin(), a[:, 2].cos(), a[:, 2].sin()
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack((o, z, z, z, cx, -sx, z, sx, cx), -1).reshape(n, 3, 3)
    Ry = torch.stack((cy, z, sy, z, o, z, -sy, z, cy), -1).reshape(n, 3, 3)
    Rz = torch.stack((cz, -sz, z, sz, cz, z, z, z, o), -1).reshape(n, 3, 3)
    return Rz @ Ry @ Rx
