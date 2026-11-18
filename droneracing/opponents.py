"""Scripted opponent drones for multi-drone racing (no camera, no RL): they follow the gate-centre line at their own speed.

Opponents are kinematic (analytic): position = point on the polyline start -> gate 0 -> gate 1 ... plus a fixed lateral / vertical
offset, advanced at a ramped-up speed. Only our drone reacts: touching an opponent ends our episode (``hit``), and the policy sees the
nearest ``cfg.opp_obs`` opponents (relative position / velocity in the body frame, only when inside the camera field of view, with
perception noise growing with distance) -- the layout a detector + tracker on the real camera could deliver. The critic sees the clean state.
"""
from __future__ import annotations

import math

import torch

from .quad import quat_to_rot

OPP_FEAT = 7  # flag, rel. position (3) / 10, rel. velocity (3) / 10, in the body frame


class Opponents:
    def __init__(self, env):
        self.env, cfg = env, env.cfg
        self.K, self.n, self.dev = cfg.opp_max, env.n, env.device
        f = dict(device=self.dev)
        P = env.gate_pos.shape[1]  # start + all gate slots except the landing slot
        self.pts = torch.zeros(self.n, P, 3, **f)
        self.cum = torch.zeros(self.n, P, **f)
        self.total = torch.zeros(self.n, **f)
        self.active = torch.zeros(self.n, self.K, dtype=torch.bool, **f)
        self.s = torch.zeros(self.n, self.K, **f)
        self.v = torch.zeros(self.n, self.K, **f)
        self.vmax = torch.ones(self.n, self.K, **f)
        self.lat = torch.zeros(self.n, self.K, **f)
        self.dz = torch.zeros(self.n, self.K, **f)
        self.pos = torch.zeros(self.n, self.K, 3, **f)
        self.vel = torch.zeros(self.n, self.K, 3, **f)
        self.yaw = torch.zeros(self.n, self.K, **f)
        self.count = torch.zeros(self.n, dtype=torch.long, **f)

    # ---- episode start ------------------------------------------------------------------
    def reset(self, ids: torch.Tensor, start: torch.Tensor, gidx0: torch.Tensor) -> None:
        env, cfg, g, d = self.env, self.env.cfg, self.env.gen, self.dev
        k, K = len(ids), self.K
        ng = env.n_gates[ids]
        gp = env.gate_pos[ids][:, :-1]  # drop the landing slot
        G = gp.shape[1]
        slot = torch.arange(G, device=d)[None]
        gp = torch.where((slot < ng[:, None])[..., None], gp, gp[torch.arange(k, device=d), (ng - 1).clamp_min(0)][:, None])  # padding = last gate
        pts = torch.cat((start[:, None], gp), 1)
        seg = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)
        cum = torch.cat((torch.zeros(k, 1, device=d), seg.cumsum(-1)), 1)
        self.pts[ids] = pts
        self.cum[ids] = cum
        self.total[ids] = cum[:, -1]
        n_opp = torch.randint(1, K + 1, (k,), generator=g, device=d)
        n_opp = torch.where(torch.rand(k, generator=g, device=d) < cfg.opp_prob, n_opp, torch.zeros_like(n_opp))
        self.count[ids] = n_opp
        self.active[ids] = torch.arange(K, device=d)[None] < n_opp[:, None]
        lo, hi = cfg.opp_speed
        self.vmax[ids] = lo + (hi - lo) * torch.rand(k, K, generator=g, device=d)
        s_ref = cum[torch.arange(k, device=d), gidx0.clamp(max=G)]
        U = lambda *sh: torch.rand(*sh, generator=g, device=d)  # noqa: E731
        wide = U(k, K) < 0.5  # start-grid style: beside / behind us with >= 1 m lateral gap; else on our line, >= 3.5 m ahead or behind
        ahead = U(k, K) < 0.5
        s_wide = s_ref[:, None] - 4.0 + 7.0 * U(k, K)
        s_line = s_ref[:, None] + torch.where(ahead, 3.5 + 6.0 * U(k, K), -3.5 - 4.0 * U(k, K))
        self.s[ids] = torch.where(wide, s_wide, s_line)
        mid = (gidx0 > 0)[:, None]
        self.v[ids] = torch.where(mid, self.vmax[ids], torch.zeros(k, K, device=d))
        sign = torch.where(U(k, K) < 0.5, -1.0, 1.0)
        self.lat[ids] = sign * torch.where(wide, 1.0 + 0.8 * U(k, K), 0.15 + 0.35 * U(k, K))
        self.dz[ids] = (U(k, K) - 0.5) * 0.5
        self._place(ids)

    # ---- motion -------------------------------------------------------------------------
    def _place(self, ids: torch.Tensor) -> None:
        cum, pts, s = self.cum[ids], self.pts[ids], self.s[ids]
        k, K = s.shape
        sc = torch.minimum(s, self.total[ids][:, None])
        i = (torch.searchsorted(cum.contiguous(), sc.contiguous()) - 1).clamp(0, cum.shape[1] - 2)
        b = torch.arange(k, device=self.dev)[:, None]
        p0, p1 = pts[b, i], pts[b, i + 1]
        seglen = (cum[b, i + 1] - cum[b, i]).clamp_min(1e-6)
        fr = ((sc - cum[b, i]) / seglen).clamp(max=1.0)[..., None]  # s < 0: extrapolate backwards along the first segment
        pos = p0 + (p1 - p0) * fr
        dirv = p1 - p0
        dirv = dirv / dirv.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        left = torch.stack((-dirv[..., 1], dirv[..., 0], torch.zeros_like(dirv[..., 0])), -1)
        left = left / left[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pos = pos + left * self.lat[ids][..., None]
        pos[..., 2] += self.dz[ids]
        self.pos[ids] = pos
        self.vel[ids] = dirv * self.v[ids][..., None]
        self.yaw[ids] = torch.atan2(dirv[..., 1], dirv[..., 0])

    def step(self, dt: float) -> None:
        if self.K == 0:
            return
        self.v = torch.minimum(self.vmax, self.v + 4.0 * dt)  # 4 m/s^2 ramp-up
        self.s = self.s + self.v * dt
        self.active &= self.s < (self.total[:, None] + 2.0)  # finished opponents leave the track
        self._place(self.env.arange)

    # ---- interaction ----------------------------------------------------------------------
    def dist(self) -> torch.Tensor:
        """Centre distance of every opponent to our drone [N,K]; inactive ones are far away."""
        d = (self.pos - self.env.quad.pos[:, None]).norm(dim=-1)
        return torch.where(self.active, d, torch.full_like(d, 1e3))

    def hit(self) -> torch.Tensor:
        cfg = self.env.cfg
        return (self.dist() < (self.env.quad.radius[:, None] + cfg.opp_radius)).any(-1)

    def obs(self, noisy: bool) -> torch.Tensor:
        """[N, opp_obs * OPP_FEAT] nearest-first features. noisy: field-of-view gate, range, miss rate and distance-scaled noise."""
        env, cfg, q = self.env, self.env.cfg, self.env.quad
        Ko = cfg.opp_obs
        d = self.dist()
        order = d.argsort(dim=-1)[:, :Ko]
        b = env.arange[:, None]
        rel = self.pos[b, order] - q.pos[:, None]
        rv = self.vel[b, order] - q.vel[:, None]
        dd = d[b, order]
        ok = dd < cfg.opp_range
        Rt = quat_to_rot(q.quat).transpose(1, 2)
        relb = (Rt[:, None] @ rel[..., None]).squeeze(-1)
        rvb = (Rt[:, None] @ rv[..., None]).squeeze(-1)
        if noisy and cfg.sensors.scale > 0:
            sn = cfg.sensors
            s_, c_ = math.sin(cfg.cam_tilt), math.cos(cfg.cam_tilt)
            depth = relb[..., 0] * c_ + relb[..., 2] * s_
            vert = -relb[..., 0] * s_ + relb[..., 2] * c_
            fov = (depth > 0.3) & (torch.atan2(relb[..., 1].abs(), depth.clamp_min(1e-3)) < sn.fov_h) & (torch.atan2(vert.abs(), depth.clamp_min(1e-3)) < sn.fov_v)
            ok = ok & fov & (torch.rand_like(dd) < cfg.opp_detect)
            sig = cfg.opp_noise[0] + cfg.opp_noise[1] * dd
            relb = relb + torch.randn_like(relb) * sig[..., None]
            rvb = rvb + torch.randn_like(rvb) * (cfg.opp_noise[2] * (1.0 + 0.1 * dd))[..., None]
        feat = torch.cat((ok.float()[..., None], (relb / 10.0).clamp(-3, 3), (rvb / 10.0).clamp(-3, 3)), -1) * ok.float()[..., None]
        return feat.reshape(env.n, -1)
