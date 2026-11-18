"""Real-camera effects for detector training at racing speed (IMX219, rolling shutter, up to 30 m/s, 14 rad/s).

Applied after ``percep_aug.realistic_augment`` (appearance). Every effect that moves image content also moves the corner labels,
so the labels stay consistent:
1. Rolling shutter: each row is captured at t_r = (row/H - 0.5) * T_readout. Rotation shifts the row by f * omega * t_r;
   forward motion scales it about the image centre by (1 + v_fwd * t_r / Z) (Z from the gate size in the image).
2. Motion blur over the exposure: rotation flow (uniform) + forward zoom flow (radial, grows with v / Z).
3. Short exposure: the signal is scaled down and amplified again by the gain; shot noise and read noise grow with the gain;
   10-bit quantization and black level.
4. Lens edge: after fisheye undistortion the image edges are upsampled; blur grows with the radius.
5. Corrupt band (radio interference, MonoRace): rows replaced by noise or a smeared row; corners in the band lose their label,
   so the detector learns to find no gate there.
The body rate and the speed of the recorded frames are zero (frames were placed): they are sampled here.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _u(n, lo, hi, dev, gen):
    return lo + (hi - lo) * torch.rand(n, generator=gen, device=dev)


def gate_depth(px: torch.Tensor, sv: torch.Tensor, f: float) -> torch.Tensor:
    """px [B,S,4,2] corner pixels, sv [B,S] valid slots -> depth of the nearest gate [B] from its image size (1.5 m opening)."""
    side = (px.amax(2) - px.amin(2)).amax(-1)  # [B,S]
    side = torch.where(sv, side, torch.zeros_like(side))
    s = side.amax(1).clamp_min(4.0)
    return (f * 1.5 / s).clamp(0.8, 40.0)


@torch.no_grad()
def realcam(img: torch.Tensor, px: torch.Tensor, vis: torch.Tensor, sv: torch.Tensor, hfov_deg: float, gen: torch.Generator | None = None,
            rs_ms_max: float = 30.0, rs_prob: float = 0.8, exp_ms: tuple = (0.5, 4.0), v_max: float = 30.0, w_sigma: float = 4.0,
            dark_prob: float = 0.4, corrupt_prob: float = 0.15, edge_prob: float = 0.5, win: torch.Tensor | None = None,
            full_hw: tuple | None = None):
    """img [B,3,H,W] 0..1, px [B,S,4,2] px, vis [B,S,4] bool, sv [B,S] bool -> (img, px, vis, sv).
    win [B,3] (x0, y0, side) and full_hw (H_full, W_full): the image is an adaptive crop of a full frame (collect_percep --crop-out);
    the focal length, the image centre (focus of expansion) and the row capture time then follow the full frame."""
    B, C, H, W = img.shape
    dev, dt = img.device, img.dtype
    if win is None:
        f = torch.full((B,), (W / 2) / math.tan(math.radians(hfov_deg) / 2), device=dev, dtype=dt)
        cx, cy = torch.full((B,), W / 2, device=dev, dtype=dt), torch.full((B,), H / 2, device=dev, dtype=dt)
        ry0, rys, Hf = torch.zeros(B, device=dev, dtype=dt), torch.ones(B, device=dev, dtype=dt), float(H)
    else:
        Hf, Wf = float(full_hw[0]), float(full_hw[1])
        win = win.to(dev, dt)
        sc_w = W / win[:, 2]  # crop px per full-frame px
        f = (Wf / 2) / math.tan(math.radians(hfov_deg) / 2) * sc_w
        cx, cy = (Wf / 2 - win[:, 0]) * sc_w, (Hf / 2 - win[:, 1]) * sc_w
        ry0, rys = win[:, 1], 1.0 / sc_w  # full-frame row of crop row r: ry0 + (r + 0.5) * rys
    f_, cx_, cy_ = f[:, None, None], cx[:, None, None], cy[:, None, None]
    om = (torch.randn(B, 3, generator=gen, device=dev) * w_sigma).clamp(-14, 14)  # camera-frame body rate [rad/s]
    v = _u(B, 0.0, v_max, dev, gen) * (torch.rand(B, generator=gen, device=dev) < 0.85).float()  # forward speed [m/s]
    side = torch.where(sv, (px.amax(2) - px.amin(2)).amax(-1), torch.zeros(px.shape[:2], device=dev, dtype=dt)).amax(1).clamp_min(4.0)
    Z = (f * 1.5 / side).clamp(0.8, 40.0)  # depth of the nearest gate from its image size (1.5 m opening)
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt), torch.arange(W, device=dev, dtype=dt), indexing="ij")

    # 1. rolling shutter (inverse map: output row r samples the source at the motion-free position)
    T = _u(B, 0.0, rs_ms_max / 1000.0, dev, gen) * (torch.rand(B, generator=gen, device=dev) < rs_prob).float()
    row_t = lambda y: ((ry0[:, None, None] + y * rys[:, None, None]) / Hf - 0.5) * T[:, None, None]  # noqa: E731
    tr = row_t(ys[None] + 0.5)  # [B,H,W]
    sc = 1 + (v / Z)[:, None, None] * tr
    du = f_ * -om[:, 1, None, None] * tr
    dv = f_ * om[:, 0, None, None] * tr
    sx = (xs[None] - cx_ - du) / sc + cx_
    sy = (ys[None] - cy_ - dv) / sc + cy_
    grid = torch.stack(((sx + 0.5) / W * 2 - 1, (sy + 0.5) / H * 2 - 1), -1)
    img = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    t_c = row_t(px[..., 1])  # label: the corner moves with the time of its row
    s_c = 1 + (v / Z)[:, None, None] * t_c
    px = torch.stack(((px[..., 0] - cx_) * s_c + cx_ + f_ * -om[:, 1, None, None] * t_c,
                      (px[..., 1] - cy_) * s_c + cy_ + f_ * om[:, 0, None, None] * t_c), -1)

    # 2. motion blur over the exposure (5 taps): rotation shift + zoom about the centre
    Te = _u(B, exp_ms[0] / 1000.0, exp_ms[1] / 1000.0, dev, gen)
    acc = torch.zeros_like(img)
    for t in torch.linspace(-0.5, 0.5, 5, device=dev):
        k = (1 + (v / Z) * Te * t)[:, None, None]
        bx = (xs[None] - cx_ - f_ * -om[:, 1, None, None] * Te[:, None, None] * t) / k + cx_
        by = (ys[None] - cy_ - f_ * om[:, 0, None, None] * Te[:, None, None] * t) / k + cy_
        g = torch.stack(((bx + 0.5) / W * 2 - 1, (by + 0.5) / H * 2 - 1), -1)
        acc = acc + F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=False)
    img = acc / 5

    # 3-4. short exposure and lens edge
    img = exposure_edge(img, gen, dark_prob, edge_prob)

    # 5. corrupt band: noise or a smeared row over 10-60 % of the rows; labels inside the band are removed
    hit = torch.rand(B, generator=gen, device=dev) < corrupt_prob
    vis, sv = vis.clone(), sv.clone()
    for b in torch.nonzero(hit).flatten().tolist():
        h = int(H * float(_u(1, 0.1, 0.6, dev, gen)))
        y0 = int((H - h) * float(torch.rand(1, generator=gen, device=dev)))
        if float(torch.rand(1, generator=gen, device=dev)) < 0.5:
            img[b, :, y0:y0 + h] = torch.rand(C, h, W, generator=gen, device=dev)
        else:
            img[b, :, y0:y0 + h] = img[b, :, y0:y0 + 1].expand(C, h, W)
        inb = (px[b, ..., 1] >= y0) & (px[b, ..., 1] < y0 + h)
        vis[b] = vis[b] & ~inb
        ctr = px[b, :, :, 1].mean(-1)
        sv[b] = sv[b] & ~((ctr >= y0) & (ctr < y0 + h))
    return img.clamp(0, 1), px, vis, sv


@torch.no_grad()
def exposure_edge(img: torch.Tensor, gen: torch.Generator | None = None, dark_prob: float = 0.4, edge_prob: float = 0.5) -> torch.Tensor:
    """Short exposure (dark signal amplified by the gain: shot + read noise, black level, 10-bit) and lens-edge blur. No geometry change."""
    B, C, H, W = img.shape
    dev, dt = img.device, img.dtype
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt), torch.arange(W, device=dev, dtype=dt), indexing="ij")
    cx, cy = W / 2, H / 2
    # short exposure: dark signal, gain, shot + read noise, black level, 10-bit quantization
    dark = torch.rand(B, generator=gen, device=dev) < dark_prob
    a = torch.where(dark, _u(B, 0.12, 0.5, dev, gen), torch.ones(B, device=dev))[:, None, None, None]  # light reaching the sensor
    full = 1023.0 * a
    sig = img * full
    sig = sig + torch.randn(sig.shape, generator=gen, device=dev) * sig.clamp_min(0).sqrt() * 0.6 + torch.randn(sig.shape, generator=gen, device=dev) * 2.5
    sig = (sig + 16.0).round().clamp(0, 1023) - 16.0
    img = (sig / full).clamp(0, 1)

    # lens edge blur (undistorted fisheye): blend with a blurred copy, weight grows with the radius
    r = (((xs - cx) / cx) ** 2 + ((ys - cy) / cy) ** 2).sqrt()[None, None] / 1.414
    e = ((torch.rand(B, generator=gen, device=dev) < edge_prob).float() * _u(B, 0.3, 1.0, dev, gen))[:, None, None, None]
    blur = F.avg_pool2d(F.pad(img, (2, 2, 2, 2), mode="replicate"), 5, 1)
    img = img * (1 - e * r**2) + blur * (e * r**2)

    return img
