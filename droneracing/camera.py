"""Analytic camera model for fast student development (no Isaac, no GPU rasterizer).

Ray-casts the next gates of every environment: gate frame ring, the marker panels above each
gate (turn colour + climb stripe + fork panel + finish), ground and sky. It is a stand-in for the
Isaac TiledCamera so that the RGB+IMU student, DAgger loop and augmentations can be built and
tested anywhere; the final student must still be validated on Isaac renders.

Camera: body-mounted, pitched up by ``cam_tilt`` about body y, pinhole with ``hfov``.
Colour code (same as droneracing.scenes): turn L blue / S green / R orange / U yellow,
climb stripe white (up) / grey (level) / black (down), fork panel purple + white square on the
correct side, finish magenta. Gate frames share one colour (randomized per episode if wanted).
"""
from __future__ import annotations

import math

import torch

from .quad import quat_to_rot

TURN_RGB = torch.tensor([[0.10, 0.45, 1.00], [0.10, 0.85, 0.30], [1.00, 0.55, 0.10], [0.95, 0.90, 0.15]])
CLIMB_RGB = torch.tensor([[0.05, 0.05, 0.05], [0.50, 0.50, 0.50], [1.00, 1.00, 1.00]])
FORK_RGB = torch.tensor([0.55, 0.20, 0.80])
FINISH_RGB = torch.tensor([1.00, 0.10, 0.80])
GATE_RGB = torch.tensor([1.00, 0.35, 0.10])
GROUND_RGB = torch.tensor([0.30, 0.34, 0.30])
SKY_RGB = torch.tensor([0.62, 0.72, 0.85])


class AnalyticCamera:
    def __init__(self, env, height: int = 48, width: int = 64, hfov_deg: float = 110.0, far: float = 40.0, n_gates: int = 3):
        self.env, self.H, self.W, self.far, self.K = env, height, width, far, n_gates
        dev = env.device
        tx = math.tan(math.radians(hfov_deg) / 2)
        ty = tx * height / width
        xs = (torch.arange(width, device=dev) + 0.5) / width * 2 - 1  # left -> right
        ys = 1 - (torch.arange(height, device=dev) + 0.5) / height * 2  # top -> bottom
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.px = (gx * tx).reshape(-1)  # [P]
        self.py = (gy * ty).reshape(-1)
        self.cols = {k: v.to(dev) for k, v in dict(turn=TURN_RGB, climb=CLIMB_RGB, fork=FORK_RGB, fin=FINISH_RGB,
                                                   gate=GATE_RGB, ground=GROUND_RGB, sky=SKY_RGB).items()}

    @torch.no_grad()
    def render(self) -> torch.Tensor:
        env, q, cfg = self.env, self.env.quad, self.env.cfg
        n, P, K = env.n, self.px.numel(), self.K
        R = quat_to_rot(q.quat)
        t = cfg.cam_tilt
        fwd = R[:, :, 0] * math.cos(t) + R[:, :, 2] * math.sin(t)
        up = -R[:, :, 0] * math.sin(t) + R[:, :, 2] * math.cos(t)
        right = -R[:, :, 1]
        d = fwd[:, None, :] + self.px[None, :, None] * right[:, None, :] + self.py[None, :, None] * up[:, None, :]
        d = d / d.norm(dim=-1, keepdim=True)  # [n, P, 3]

        best_t = torch.full((n, P), self.far, device=env.device)
        img = torch.zeros(n, P, 3, device=env.device)
        # sky / ground
        dz = d[..., 2]
        t_ground = torch.where(dz < -1e-4, -q.pos[:, 2:3] / dz, torch.full_like(dz, 1e9))
        ground_hit = (t_ground > 0) & (t_ground < self.far)
        img[:] = self.cols["sky"]
        gx = q.pos[:, None, :2] + t_ground[..., None] * d[..., :2]
        checker = ((torch.floor(gx[..., 0] / 2) + torch.floor(gx[..., 1] / 2)) % 2) * 0.06
        gcol = self.cols["ground"] + checker[..., None]
        img = torch.where(ground_hit[..., None], gcol, img)
        best_t = torch.where(ground_hit, t_ground, best_t)

        last = env._last_slot()
        half = 0.5 * env.gate_size[:, None]
        fr = cfg.track.gate_frame
        for j in range(K):
            idx = torch.minimum(env.gate_idx + j, last)
            c, nrm, rgt = env._gate_frame(idx)
            zhat = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(c)
            denom = (d * nrm[:, None, :]).sum(-1)
            tt = ((c - q.pos) * nrm).sum(-1, keepdim=True) / denom.where(denom.abs() > 1e-6, torch.full_like(denom, 1e-6))
            h = q.pos[:, None, :] + tt[..., None] * d
            lat = ((h - c[:, None, :]) * rgt[:, None, :]).sum(-1)
            ver = ((h - c[:, None, :]) * zhat[:, None, :]).sum(-1)
            ok = (tt > 0.05) & (tt < best_t) & (idx < env.n_gates + (1 if cfg.landing else 0))[:, None]
            ring = ok & (torch.maximum(lat.abs(), ver.abs()) > half) & (torch.maximum(lat.abs(), ver.abs()) < half + fr)
            top = half + fr
            token_ok = ok & (idx < env.n_gates)[:, None] & cfg.markers_visible if hasattr(cfg, "markers_visible") else ok & (idx < env.n_gates)[:, None]
            panel = token_ok & (lat.abs() < 0.28) & (ver > top + 0.30) & (ver < top + 0.80)
            stripe = token_ok & (lat.abs() < 0.28) & (ver > top + 0.15) & (ver <= top + 0.30)
            turn = env.m_turn[env.arange, idx]
            climb = env.m_climb[env.arange, idx]
            fin = (idx == env.n_gates - 1)
            pcol = self.cols["turn"][turn][:, None, :].expand(n, P, 3)
            pcol = torch.where(fin[:, None, None], self.cols["fin"].expand(n, P, 3), pcol)
            scol = self.cols["climb"][climb][:, None, :].expand(n, P, 3)
            # fork marker lives on the gate before the fork gate: slot idx carries it if m_fork != 0
            fk = env.m_fork[env.arange, idx]
            fpanel = token_ok & (fk != 0)[:, None] & (lat.abs() < 0.35) & (ver > top + 0.9) & (ver < top + 1.4)
            side = torch.where(fk == 1, -1.0, 1.0)[:, None]  # left of travel = -right
            fsq = fpanel & ((lat - side * 0.2).abs() < 0.11) & ((ver - (top + 1.15)).abs() < 0.11)
            for m, col in ((ring, self.cols["gate"].expand(n, P, 3)), (panel, pcol), (stripe, scol),
                           (fpanel, self.cols["fork"].expand(n, P, 3)), (fsq, torch.ones(n, P, 3, device=env.device))):
                img = torch.where(m[..., None], col, img)
                best_t = torch.where(m, tt.expand(n, P), best_t)
        # haze with distance
        fog = (best_t / self.far).clamp(0, 1)[..., None] * 0.35
        img = img * (1 - fog) + self.cols["sky"] * fog
        return img.reshape(n, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()


def augment(img: torch.Tensor, omega: torch.Tensor, scale: float = 1.0, gen: torch.Generator | None = None) -> torch.Tensor:
    """Photometric / sensor corruption of rendered frames (domain randomization of the camera).

    exposure + per-channel gain, additive noise, motion blur growing with body rate, and corrupted
    frames (MonoRace handled up to 50 % corrupted images). ``scale`` = 0 returns the input.
    """
    if scale <= 0:
        return img
    n = img.shape[0]
    dev = img.device
    r = lambda *s: torch.rand(*s, generator=gen, device=dev)  # noqa: E731
    gain = (1.0 + (r(n, 1, 1, 1) - 0.5) * 0.8 * scale) * (1.0 + (r(n, 3, 1, 1) - 0.5) * 0.2 * scale)
    out = (img * gain + (r(n, 1, 1, 1) - 0.5) * 0.1 * scale)
    out = out + (r(n, 3, *img.shape[2:]) - 0.5) * 0.08 * scale
    rate = omega.norm(dim=-1)[:, None, None, None]
    blur = (rate > 6.0).float()
    out = (1 - 0.5 * blur) * out + 0.5 * blur * torch.roll(out, 1, dims=3)
    bad = (r(n, 1, 1, 1) < 0.08 * scale).float()
    out = (1 - bad) * out + bad * r(n, 3, *img.shape[2:])
    return out.clamp(0, 1)
