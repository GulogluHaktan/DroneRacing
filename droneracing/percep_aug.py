"""Realistic camera augmentation for GateNet: background / lighting / motion-blur / sensor-noise randomization.

The Isaac renders have a flat sky and a dark grid floor. A real hall has ceiling lights, banners, coloured floors,
exposure changes, motion blur and sensor noise. Only appearance is changed (no geometry), so the simulator's
corner labels stay valid. Sky and gate pixels are found from the clean render by colour (sky = near-white,
unsaturated; gate = violet/purple), which avoids storing masks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _u(n, lo, hi, dev, gen, *shape):
    return lo + (hi - lo) * torch.rand(n, *shape, generator=gen, device=dev)


def _lowfreq(n, h, w, gh, gw, dev, gen):
    return F.interpolate(torch.rand(n, 1, gh, gw, generator=gen, device=dev) * 2 - 1, size=(h, w), mode="bilinear", align_corners=False)


def motion_blur(img, flow_px, taps=5):
    """Average ``taps`` copies shifted along the per-sample image-space flow [B,2] (pixels over the exposure)."""
    B, _, H, W = img.shape
    out = torch.zeros_like(img)
    base = F.affine_grid(torch.eye(2, 3, device=img.device).expand(B, 2, 3), img.shape, align_corners=False)
    for t in torch.linspace(-0.5, 0.5, taps):
        shift = torch.stack((flow_px[:, 0] * 2 / W, flow_px[:, 1] * 2 / H), -1) * t
        out = out + F.grid_sample(img, base + shift[:, None, None, :], mode="bilinear", padding_mode="border", align_corners=False)
    return out / taps


def realistic_augment(img: torch.Tensor, scale: float = 1.0, flow_px: torch.Tensor | None = None,
                      gen: torch.Generator | None = None) -> torch.Tensor:
    if scale <= 0:
        return img
    B, _, H, W = img.shape
    dev = img.device
    sat = img.amax(1) - img.amin(1)
    sky = ((sat < 0.035) & (img.mean(1) > 0.86))[:, None]
    gate = ((img[:, 2] > img[:, 0] + 0.12) & (img[:, 2] > img[:, 1] + 0.15))[:, None]
    ground = ~sky & ~gate
    ys = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1)

    # --- background: sky replaced by gradients + lights / dark clutter
    base = _u(B, 0.05, 0.95, dev, gen, 1, 1, 1)
    c1 = (base * torch.exp(_u(B, -0.3, 0.3, dev, gen, 3, 1, 1))).clamp(0, 1)
    base2 = (base + _u(B, -0.35, 0.35, dev, gen, 1, 1, 1)).clamp(0.02, 1)
    c2 = (base2 * torch.exp(_u(B, -0.3, 0.3, dev, gen, 3, 1, 1))).clamp(0, 1)
    bg = c1 * (1 - ys) + c2 * ys
    bg = bg + _lowfreq(B, H, W, 3, 4, dev, gen) * 0.12 + _lowfreq(B, H, W, 12, 16, dev, gen) * 0.05
    xs = torch.linspace(0, 1, W, device=dev).view(1, 1, 1, W)
    for _ in range(3):
        cx, cy = _u(B, 0, 1, dev, gen, 1, 1, 1), _u(B, 0, 0.7, dev, gen, 1, 1, 1)
        sx, sy = _u(B, 0.02, 0.18, dev, gen, 1, 1, 1), _u(B, 0.01, 0.08, dev, gen, 1, 1, 1)
        amp = _u(B, -0.6, 0.8, dev, gen, 1, 1, 1) * (torch.rand(B, 1, 1, 1, generator=gen, device=dev) < 0.6).float()
        bg = bg + amp * torch.exp(-((xs - cx) ** 2 / (2 * sx**2) + (ys - cy) ** 2 / (2 * sy**2)))
    out = torch.where(sky, bg.clamp(0, 1), img)

    # --- floor / other surfaces: tint, brightness gradient, texture
    tint = torch.exp(_u(B, -0.5, 0.5, dev, gen, 3, 1, 1) + _u(B, -0.4, 0.4, dev, gen, 1, 1, 1))
    grad = 1 + _u(B, -0.4, 0.4, dev, gen, 1, 1, 1) * (ys - 0.5) * 2
    tex = 1 + _lowfreq(B, H, W, 8, 10, dev, gen) * 0.15 + (torch.rand(B, 1, H, W, generator=gen, device=dev) - 0.5) * 0.1
    lift = _u(B, 0.0, 0.35, dev, gen, 1, 1, 1) * (torch.rand(B, 1, 1, 1, generator=gen, device=dev) < 0.5).float()  # bright floors
    out = torch.where(ground, (out * tint * grad * tex + lift).clamp(0, 1), out)

    # --- global illumination / camera response
    gain = _u(B, 0.6, 1.5, dev, gen, 1, 1, 1) * torch.exp(_u(B, -0.12, 0.12, dev, gen, 3, 1, 1))
    gamma = _u(B, 0.7, 1.5, dev, gen, 1, 1, 1)
    out = (out.clamp(1e-4, 1) ** gamma * gain).clamp(0, 1)
    r2 = ((xs - 0.5) ** 2 + (ys - 0.5) ** 2) * 2
    out = out * (1 - _u(B, 0.0, 0.5, dev, gen, 1, 1, 1) * r2)

    # --- motion blur (exposure x image-space flow) and slight defocus
    if flow_px is None:
        mag = torch.rand(B, generator=gen, device=dev).pow(2.0) * 9.0 * scale
        ang = torch.rand(B, generator=gen, device=dev) * 6.2832
        horiz = torch.rand(B, generator=gen, device=dev) < 0.6
        ang = torch.where(horiz, ang * 0.15, ang)
        flow_px = torch.stack((mag * torch.cos(ang), mag * torch.sin(ang)), -1)
    out = motion_blur(out, flow_px)
    k = (torch.rand(B, 1, 1, 1, generator=gen, device=dev) < 0.3).float()
    out = (1 - k) * out + k * F.avg_pool2d(F.pad(out, (1, 1, 1, 1), mode="replicate"), 3, 1)

    # --- sensor noise (signal dependent) + rare salt & pepper
    sigma = _u(B, 0.0, 0.05, dev, gen, 1, 1, 1) * scale
    out = out + torch.randn(out.shape, generator=gen, device=dev) * sigma * (0.4 + out)
    sp = (torch.rand(B, 1, H, W, generator=gen, device=dev) < 0.0015 * scale).float()
    out = (1 - sp) * out + sp * (torch.rand(B, 1, H, W, generator=gen, device=dev) > 0.5).float()
    return out.clamp(0, 1)
