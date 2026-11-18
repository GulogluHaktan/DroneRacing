"""Batched gate-racing environment.

Every reset samples a new vehicle (mass / max thrust / size / drag / lags) and a
new rule-based track (gate count & spacing, obstacles). One episode = one lap.

Optional task features (EnvCfg flags):
* ``markers``: privileged marker tokens in the observation (teacher) and forks
  (correct gate + decoy; flying through the decoy is a crash);
* ``landing``: after the last gate the vehicle must land (slow + low) in a zone.
``imu_obs()`` provides noisy gyro/accelerometer readings for sensor-based students.
"""
from __future__ import annotations

import math

import torch

from .config import EnvCfg
from .quad import GRAVITY, QuadBatch, quat_exp, quat_to_rot
from .track import generate_tracks, track_length
from .opponents import OPP_FEAT, Opponents

ACT_DIM = 4
HIST_DIM = 10  # body velocity (3) + gyro (3) + previous action (4)
TOKEN_DIM = 11  # turn(4: L/S/R/U-turn) + climb(3) + fork(3) + finish(1)


class RaceEnv:
    def __init__(self, cfg: EnvCfg, seed: int = 0, quad_factory=None):
        self.cfg = cfg
        self.n, self.device = cfg.n_envs, cfg.device
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        factory = quad_factory or QuadBatch
        self.quad = factory(self.n, self.device, cfg.vehicle, self.gen)
        G, M = cfg.track.max_gates, cfg.track.max_obstacles
        f = dict(device=self.device)
        self.gate_pos = torch.zeros(self.n, G + 1, 3, **f)  # slot n_gates = landing point
        self.gate_yaw = torch.zeros(self.n, G + 1, **f)
        self.decoy_pos = torch.zeros(self.n, G + 1, 3, **f)
        self.decoy_on = torch.zeros(self.n, G + 1, dtype=torch.bool, **f)
        self.m_turn = torch.zeros(self.n, G + 1, dtype=torch.long, **f)
        self.m_climb = torch.zeros(self.n, G + 1, dtype=torch.long, **f)
        self.m_fork = torch.zeros(self.n, G + 1, dtype=torch.long, **f)
        self.gate_size = torch.ones(self.n, **f)
        self.n_gates = torch.ones(self.n, dtype=torch.long, **f)
        self.obs_xy = torch.zeros(self.n, M, 2, **f)
        self.obs_r = torch.zeros(self.n, M, **f)
        self.obs_on = torch.zeros(self.n, M, dtype=torch.bool, **f)
        self.track_len = torch.zeros(self.n, **f)
        self.soft = torch.zeros(self.n, dtype=torch.bool, **f)  # soft gate-frame collisions in this episode
        self.hall = torch.tensor([0.0, 0.0, 1e4, 1e4], **f).repeat(self.n, 1)  # (cx, cy, half-length, half-width) of the hall walls
        self.gate_idx = torch.zeros(self.n, dtype=torch.long, **f)
        self.steps = torch.zeros(self.n, dtype=torch.long, **f)
        self.cam_off = torch.randint(0, 64, (self.n,), device=self.device)
        self.max_steps = torch.ones(self.n, dtype=torch.long, **f)
        self.prev_action = torch.zeros(self.n, ACT_DIM, **f)
        self.ep_return = torch.zeros(self.n, **f)
        self.gyro_bias = torch.zeros(self.n, 3, **f)
        self.accel_bias = torch.zeros(self.n, 3, **f)
        self.arange = torch.arange(self.n, device=self.device)
        # sensing / actuation imperfections (per-episode draws, see SensorCfg)
        self.att_bias = torch.zeros(self.n, 3, **f)
        self.vel_bias = torch.zeros(self.n, 3, **f)
        self.vel_drift = torch.zeros(self.n, 3, **f)
        self.gate_bias = torch.zeros(self.n, G + 1, 3, **f)
        self.gate_yaw_bias = torch.zeros(self.n, G + 1, **f)
        self.imu_gyro_scale = torch.ones(self.n, 3, **f)
        self.imu_accel_scale = torch.ones(self.n, 3, **f)
        self.obs_delay = torch.zeros(self.n, dtype=torch.long, **f)
        self.act_delay = torch.zeros(self.n, dtype=torch.long, **f)
        self.sag = torch.zeros(self.n, **f)
        self.wind = torch.zeros(self.n, 3, **f)
        self.gust = torch.zeros(self.n, 3, **f)
        self.hold = torch.zeros(self.n, dtype=torch.bool, **f)
        self.prev_eff = torch.zeros(self.n, ACT_DIM, **f)
        self.vis_w = torch.zeros(self.n, G + 1, 3, **f)  # held world estimate of each gate (vis_obs)
        self.vis_yaw = torch.zeros(self.n, G + 1, **f)
        self.vis_err = torch.zeros(self.n, G + 1, 3, **f)  # temporally correlated estimation error of each gate (filter output)
        self.tail_mul = torch.ones(self.n, G + 1, **f)  # heavy-tail perception events (cfg.tail_prob)
        self.split_mul = torch.ones(self.n, G + 1, **f)  # perception degradation at / after a split-S (cfg.split_noise)
        self.vis_seen = torch.zeros(self.n, G + 1, dtype=torch.bool, **f)
        self.vis_age = torch.ones(self.n, G + 1, **f)  # time since the last sighting / 3 s, clipped to 1 (age_obs)
        self.odo_drift = torch.zeros(self.n, 3, **f)
        self.gate_last = None
        self._gate_part = None
        self.gate_dim = 7 + int(cfg.vis_obs)
        base = 3 + 3 + 3 + 4 + cfg.n_obs_gates * self.gate_dim + min(cfg.n_obs_obstacles, M) * 5  # vel, gravity(body), gyro, prev action
        extra = (cfg.n_obs_gates * TOKEN_DIM if cfg.markers else 0) + (1 if cfg.landing else 0)
        self.policy_dim = base + extra + (4 if cfg.policy_sees_params else 0) + cfg.obs_history * HIST_DIM + (cfg.n_obs_gates if cfg.age_obs else 0)
        self.hist = torch.zeros(self.n, max(cfg.obs_history, 1), HIST_DIM, **f)
        self._frame = torch.zeros(self.n, HIST_DIM, **f)
        self.critic_dim = base + extra + 4 + 1
        self.opp = Opponents(self) if cfg.opp_max > 0 else None
        if self.opp is not None:  # opponent block at the end of the policy and critic observations
            self.policy_dim += cfg.opp_obs * OPP_FEAT
            self.critic_dim += cfg.opp_obs * OPP_FEAT
        self.rb_res = None
        if cfg.resid_bank:  # empirical residual bank (scripts/build_resid_bank.py)
            rb = torch.load(cfg.resid_bank, map_location=self.device)
            self.rb_res = rb["res"].float()  # [T,12]: next gate (body, m), gate after (body, m), velocity (body, m/s), up vector
            self.rb_bin = rb["bin"].long()  # [T] distance bin of the next gate; -1 marks the end of a trace segment
            self.rb_edges = rb["edges"].float()
            nb = len(self.rb_edges) + 1
            lists = [torch.nonzero(self.rb_bin == b_).flatten() for b_ in range(nb)]
            allv = torch.nonzero(self.rb_bin >= 0).flatten()
            lists = [l_ if len(l_) else allv for l_ in lists]
            self.rb_cnt = torch.tensor([len(l_) for l_ in lists], device=self.device)
            self.rb_lists = torch.zeros(nb, int(self.rb_cnt.max()), dtype=torch.long, device=self.device)
            for b_, l_ in enumerate(lists):
                self.rb_lists[b_, : len(l_)] = l_
            self.rb_ptr = torch.zeros(self.n, dtype=torch.long, **f)
            self.rb_force = torch.ones(self.n, dtype=torch.bool, **f)
            self.rb_on = torch.arange(self.n, device=self.device) < round(cfg.resid_prob * self.n)  # envs that replay the bank
        self.reset_idx(self.arange)
        if self.rb_res is not None:
            self._rb_advance()
        self._update_vis(drift=False)
        self.obs_buf = torch.zeros(self.n, cfg.sensors.obs_delay[1] + 1, self.policy_dim, **f)
        pol, _ = self.compute_obs()
        self.obs_buf[:] = pol[:, None]
        self.gate_last = self._gate_part.clone()

    # ---- reset --------------------------------------------------------------------------
    def reset_idx(self, ids: torch.Tensor) -> None:
        cfg, g, d = self.cfg, self.gen, self.device
        k = len(ids)
        self.quad.sample_params(ids)
        tr = generate_tracks(k, cfg.track, self.quad.capability(ids), g, d)
        self.gate_pos[ids], self.gate_yaw[ids] = tr["gate_pos"], tr["gate_yaw"]
        self.decoy_pos[ids], self.decoy_on[ids] = tr["decoy_pos"], tr["decoy_on"]
        self.m_turn[ids], self.m_climb[ids], self.m_fork[ids] = tr["marker_turn"], tr["marker_climb"], tr["marker_fork"]
        self.gate_size[ids], self.n_gates[ids] = tr["gate_size"], tr["n_gates"]
        self.obs_xy[ids], self.obs_r[ids], self.obs_on[ids] = tr["obs_xy"], tr["obs_r"], tr["obs_on"]
        length = track_length(tr)
        self.track_len[ids] = length
        self.hall[ids] = tr["hall"]
        self.soft[ids] = torch.rand(k, generator=g, device=d) < cfg.soft_frac

        yaw0 = tr["gate_yaw"][:, 0]
        direction = torch.stack((torch.cos(yaw0), torch.sin(yaw0)), -1)
        start = tr["gate_pos"][:, 0].clone()
        start[:, :2] -= direction * tr["start_dist"][:, None]
        start[:, 2] = (start[:, 2] + (torch.rand(k, generator=g, device=d) - 0.5) * 0.4).clamp_min(0.6)
        yaw_noise = (torch.rand(k, generator=g, device=d) - 0.5) * 0.6
        vel0 = torch.zeros(k, 3, device=d)
        gidx0 = torch.zeros(k, dtype=torch.long, device=d)
        yaw_start = yaw0 + yaw_noise
        if cfg.mid_init_prob > 0:
            # start in front of a random gate (incl. split-S / loop gates) with speed along its normal
            ng = tr["n_gates"]
            mid = (torch.rand(k, generator=g, device=d) < cfg.mid_init_prob) & (ng >= 2)
            j = (torch.rand(k, generator=g, device=d) * (ng - 1).clamp_min(1)).long() + 1
            j = torch.minimum(j, (ng - 1).clamp_min(1))
            if cfg.mid_focus and cfg.mid_focus_prob > 0:  # fast model: more fast approaches to the gates where fast flights crash
                fg = torch.tensor(cfg.mid_focus, device=d)
                pick = fg[torch.randint(0, len(fg), (k,), generator=g, device=d)]
                j = torch.where(torch.rand(k, generator=g, device=d) < cfg.mid_focus_prob, pick, j)
                j = torch.minimum(j, (ng - 1).clamp_min(1))
            bb = torch.arange(k, device=d)
            yj = tr["gate_yaw"][bb, j]
            nj = torch.stack((torch.cos(yj), torch.sin(yj), torch.zeros_like(yj)), -1)
            lo, hi = cfg.mid_init_dist
            dist = lo + (hi - lo) * torch.rand(k, 1, generator=g, device=d)
            p_mid = tr["gate_pos"][bb, j] - nj * dist
            p_mid[:, 2] = (p_mid[:, 2] + (torch.rand(k, generator=g, device=d) - 0.5) * 1.2).clamp_min(0.6)
            sp = cfg.mid_init_speed[0] + (cfg.mid_init_speed[1] - cfg.mid_init_speed[0]) * torch.rand(k, 1, generator=g, device=d)
            start = torch.where(mid[:, None], p_mid, start)
            vel0 = torch.where(mid[:, None], nj * sp, vel0)
            gidx0 = torch.where(mid, j, gidx0)
            yaw_start = torch.where(mid, yj + yaw_noise, yaw_start)
        self.quad.reset(ids, start, yaw_start, vel0)
        if self.opp is not None:
            self.opp.reset(ids, start, gidx0)

        self.gate_idx[ids] = gidx0
        self.steps[ids] = 0
        self.cam_off[ids] = torch.randint(0, 64, (len(ids),), device=self.device)
        dt_pol = cfg.dt * cfg.decimation
        t_max = (length / 3.0 + 8.0 + (6.0 if cfg.landing else 0.0)).clamp(max=cfg.max_episode_s)
        self.max_steps[ids] = (t_max / dt_pol).long()
        self.prev_action[ids] = 0.0
        self.hist[ids] = 0.0
        if self.rb_res is not None:
            self.rb_force[ids] = True
        self.vis_seen[ids] = False
        self.vis_age[ids] = 1.0
        self.vis_w[ids] = 0.0
        self.vis_yaw[ids] = 0.0
        self.vis_err[ids] = 0.0
        self.odo_drift[ids] = 0.0
        self.vel_drift[ids] = 0.0
        self.ep_return[ids] = 0.0
        self.gyro_bias[ids] = torch.randn(k, 3, generator=g, device=d) * cfg.gyro_bias
        sn = cfg.sensors
        sc = sn.scale
        G1 = self.gate_bias.shape[1]
        lo, hi = sn.bias_rand  # per-episode severity multiplier of the map / velocity biases (log-uniform): one policy for a range of map qualities
        bm = torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(k, generator=g, device=d)) if hi > lo else torch.ones(k, device=d)
        self.att_bias[ids] = torch.randn(k, 3, generator=g, device=d) * sn.att_bias * sc
        self.vel_bias[ids] = torch.randn(k, 3, generator=g, device=d) * sn.vel_bias * sc * bm[:, None]
        self.tail_mul[ids] = 1.0
        if cfg.tail_prob > 0:
            self.tail_mul[ids] = torch.where(torch.rand(k, G1, generator=g, device=d) < cfg.tail_prob, cfg.tail_mult, 1.0)
        if cfg.split_noise:  # split-S lower gate: under the previous gate (< 1.5 m in x/y), lower, flown the opposite way
            gp, gy = self.gate_pos[ids], self.gate_yaw[ids]
            low = torch.zeros(k, G1, dtype=torch.bool, device=d)
            dyaw = torch.atan2(torch.sin(gy[:, 1:] - gy[:, :-1]), torch.cos(gy[:, 1:] - gy[:, :-1])).abs()
            low[:, 1:] = ((gp[:, 1:, :2] - gp[:, :-1, :2]).norm(dim=-1) < 1.5) & (gp[:, 1:, 2] < gp[:, :-1, 2] - 1.0) & (dyaw > 2.5)
            sm = torch.ones(k, G1, device=d)
            for o, mlt in enumerate(cfg.split_noise):
                sh = torch.zeros_like(low)
                sh[:, o:] = low[:, :G1 - o]
                sm = torch.where(sh, torch.maximum(sm, torch.full_like(sm, mlt)), sm)
            self.split_mul[ids] = sm
        self.gate_bias[ids] = torch.randn(k, G1, 3, generator=g, device=d) * sn.gate_bias * sc * bm[:, None, None] * self.tail_mul[ids][..., None]
        self.gate_yaw_bias[ids] = torch.randn(k, G1, generator=g, device=d) * sn.gate_yaw_bias * sc * bm[:, None]
        if cfg.map_obs:  # approximate map: every gate is known from the start, with its map error
            self.vis_seen[ids] = True
            self.vis_w[ids] = self.gate_pos[ids] + self.gate_bias[ids]
            self.vis_yaw[ids] = self.gate_yaw[ids] + self.gate_yaw_bias[ids]
        self.imu_gyro_scale[ids] = 1.0 + torch.randn(k, 3, generator=g, device=d) * sn.gyro_scale_err * sc
        self.imu_accel_scale[ids] = 1.0 + torch.randn(k, 3, generator=g, device=d) * sn.accel_scale_err * sc
        if sc > 0:
            lo, hi = sn.obs_delay
            self.obs_delay[ids] = torch.randint(lo, hi + 1, (k,), generator=g, device=d)
            lo, hi = sn.act_delay
            self.act_delay[ids] = torch.randint(lo, hi + 1, (k,), generator=g, device=d)
        else:
            self.obs_delay[ids] = 0
            self.act_delay[ids] = 0
        self.sag[ids] = torch.rand(k, generator=g, device=d) * sn.sag * sc
        wd = torch.randn(k, 3, generator=g, device=d)
        wd = wd / wd.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.wind[ids] = wd * torch.rand(k, 1, generator=g, device=d) * sn.wind * sc
        self.gust[ids] = 0.0
        self.hold[ids] = False
        self.prev_eff[ids] = 0.0
        if self.gate_last is not None:
            self.gate_last[ids] = 0.0
        self.accel_bias[ids] = torch.randn(k, 3, generator=g, device=d) * cfg.accel_bias

    # ---- helpers ------------------------------------------------------------------------
    def _last_slot(self) -> torch.Tensor:
        return self.n_gates if self.cfg.landing else self.n_gates - 1

    def _gate_frame(self, idx: torch.Tensor, noisy: bool = False):
        yaw = self.gate_yaw[self.arange, idx]
        if noisy and self.cfg.sensors.scale > 0:
            yaw = yaw + self.gate_yaw_bias[self.arange, idx]
        normal = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), -1)
        right = torch.stack((torch.sin(yaw), -torch.cos(yaw), torch.zeros_like(yaw)), -1)
        c = self.gate_pos[self.arange, idx]
        if noisy and self.cfg.sensors.scale > 0:
            c = c + self.gate_bias[self.arange, idx]
        return c, normal, right

    def _tokens(self, idx: torch.Tensor) -> torch.Tensor:
        """Marker token of the gate in slot idx: one-hot turn, climb, fork + finish flag [N, 10]."""
        real = (idx < self.n_gates).float()[:, None]
        turn = torch.nn.functional.one_hot(self.m_turn[self.arange, idx], 4).float()
        climb = torch.nn.functional.one_hot(self.m_climb[self.arange, idx], 3).float()
        fork = torch.nn.functional.one_hot(self.m_fork[self.arange, idx], 3).float()
        finish = (idx == self.n_gates - 1).float()[:, None]
        return torch.cat((turn, climb, fork, finish), -1) * real

    def in_landing(self) -> torch.Tensor:
        return (self.gate_idx >= self.n_gates) & self.cfg.landing

    def _update_vis(self, drift: bool = True) -> None:
        """Camera-like gate knowledge: a gate becomes known when its centre is inside the field of view
        (and range); its world estimate is then held (with odometry drift) until it is seen again."""
        cfg, q, sn = self.cfg, self.quad, self.cfg.sensors
        if not cfg.vis_obs:
            return
        sc = sn.scale
        if sc > 0 and drift:
            dtp = cfg.dt * cfg.decimation
            if cfg.map_obs and sn.odo_tau > 0:
                a_ = math.exp(-dtp / sn.odo_tau)
                self.odo_drift = self.odo_drift * a_ + torch.randn_like(self.odo_drift) * sn.odo_sigma * sc * math.sqrt(1 - a_ * a_)
            else:
                self.odo_drift += torch.randn_like(self.odo_drift) * sn.odo_drift * sc * math.sqrt(dtp)
            if sn.vel_drift > 0:
                self.vel_drift = (self.vel_drift + torch.randn_like(self.vel_drift) * sn.vel_drift * sc * math.sqrt(cfg.dt * cfg.decimation)).clamp(-sn.vel_drift_cap, sn.vel_drift_cap)
        Rm = quat_to_rot(q.quat)
        Rt = Rm.transpose(1, 2)
        s, c_ = math.sin(cfg.cam_tilt), math.cos(cfg.cam_tilt)
        a_opt = torch.tensor((c_, 0.0, s), device=self.device)
        a_up = torch.tensor((-s, 0.0, c_), device=self.device)
        last = self._last_slot()
        if drift and cfg.age_obs:
            self.vis_age = (self.vis_age + cfg.dt * cfg.decimation / 3.0).clamp_max(1.0)
        if drift and cfg.corner_vis:
            from .detector_model import project_corners, _interp
            uv_, z_, ok_ = project_corners(self)
            ins_ = (uv_[..., 0] >= 0) & (uv_[..., 0] < 256) & (uv_[..., 1] >= 0) & (uv_[..., 1] < 192) & (z_ > 0.3)
            nin_ = ins_.sum(-1)  # corners in the image per slot (-1, 0, 1, 2)
            ext_ = (uv_.amax(2) - uv_.amin(2)).amax(-1)
        for j in range(cfg.n_obs_gates):
            idx = torch.minimum(self.gate_idx + j, last)
            c, _, _ = self._gate_frame(idx)
            rel_b = (Rt @ (c - q.pos).unsqueeze(-1)).squeeze(-1)
            depth = rel_b @ a_opt
            horiz = rel_b[:, 1]
            vert = rel_b @ a_up
            dist = (c - q.pos).norm(dim=-1)
            vis = (depth > 0.3) & (torch.atan2(horiz.abs(), depth.clamp_min(1e-3)) < sn.fov_h) \
                & (torch.atan2(vert.abs(), depth.clamp_min(1e-3)) < sn.fov_v) & (dist < sn.see_range)
            if cfg.cam_period > 1 and drift:  # frame rate below policy rate
                vis &= (self.steps + self.cam_off) % cfg.cam_period == 0
            if drift and cfg.corner_vis and j + 1 < nin_.shape[1]:  # measured: all 4 corners ~0.9, partial gates 4-14 %, tiny gates missed
                n_ = nin_[:, j + 1]
                cp_ = cfg.corner_vis_p
                p_ = torch.where(n_ >= 4, cp_[0], torch.where(n_ == 3, cp_[1], torch.where(n_ == 2, cp_[2], 0.0)))
                p_ = p_ * _interp(ext_[:, j + 1], [6, 10, 14, 20], [0.02, 0.35, 0.75, 1.0])
                vis &= torch.rand_like(p_) < p_
            if not vis.any():
                continue
            rho = sn.gate_err_corr ** cfg.cam_period  # AR(1) between two frames
            sig = (sn.gate_noise + sn.gate_noise_dist * dist[:, None]) * sc * (1.0 + 0.8 * j) * self.tail_mul[self.arange, idx][:, None] * self.split_mul[self.arange, idx][:, None]  # farther gate: coarser
            err = rho * self.vis_err[self.arange, idx] + math.sqrt(1 - rho * rho) * sig * torch.randn_like(c)  # AR(1): filter-like error
            noise = err
            if cfg.map_obs:  # a sighting replaces the map error by a PnP measurement
                est = c + noise + self.odo_drift
                yaw = self.gate_yaw[self.arange, idx] + torch.randn_like(dist) * sn.yaw_noise * sc
            else:
                est = c + self.gate_bias[self.arange, idx] + noise + self.odo_drift
                yaw = self.gate_yaw[self.arange, idx] + self.gate_yaw_bias[self.arange, idx] + torch.randn_like(dist) * 0.03 * sc
            m = vis
            self.vis_w[self.arange[m], idx[m]] = est[m]
            self.vis_err[self.arange[m], idx[m]] = err[m]
            self.vis_yaw[self.arange[m], idx[m]] = yaw[m]
            self.vis_seen[self.arange[m], idx[m]] = True
            self.vis_age[self.arange[m], idx[m]] = 0.0

    def _rb_advance(self) -> None:
        """Step every env one sample along its residual trace; jump to a random sample of the matching distance bin when the
        trace ends, after a reset, or when the trace's distance bin is more than one bin away from the true one."""
        c0, _, _ = self._gate_frame(torch.minimum(self.gate_idx, self._last_slot()))
        b = torch.bucketize((c0 - self.quad.pos).norm(dim=-1), self.rb_edges)
        nxt = (self.rb_ptr + 1).clamp(max=len(self.rb_bin) - 1)
        nb_ = self.rb_bin[nxt]
        bad = self.rb_force | (nb_ < 0) | ((nb_ - b).abs() > 1)
        cnt = self.rb_cnt[b]
        r = (torch.rand(self.n, generator=self.gen, device=self.device) * cnt).long().clamp(max=cnt - 1)
        self.rb_ptr = torch.where(bad, self.rb_lists[b, r], nxt)
        self.rb_force[:] = False

    # ---- observations -------------------------------------------------------------------
    def _parts(self, noisy: bool):
        """Observation blocks; ``noisy`` applies the estimator / perception errors of SensorCfg (or the residual bank)."""
        if noisy and self.rb_res is not None and self.cfg.resid_prob < 1.0:  # mix: bank on a fixed subset of envs, parametric errors elsewhere
            a, b = self._parts_one(True, True), self._parts_one(True, False)
            m = self.rb_on[:, None]
            return tuple([torch.where(m, x, y) for x, y in zip(la, lb)] for la, lb in zip(a, b))
        return self._parts_one(noisy, True)

    def _parts_one(self, noisy: bool, bank: bool):
        cfg, q, sn = self.cfg, self.quad, self.cfg.sensors
        sc = sn.scale if noisy else 0.0
        R = quat_to_rot(q.quat)
        vel, omega = q.vel, q.omega
        res = self.rb_res[self.rb_ptr] * cfg.resid_scale if (sc > 0 and bank and self.rb_res is not None) else None
        if res is not None:  # empirical residuals replace the parametric state and gate errors (IMU-side noise stays)
            omega = omega + self.gyro_bias + torch.randn_like(omega) * sn.gyro_noise_obs * sc
        elif sc > 0:
            err = self.att_bias + torch.randn_like(self.att_bias) * sn.att_noise * sc
            R = quat_to_rot(quat_exp(err)) @ R
            vel = vel + self.vel_bias + self.vel_drift + torch.randn_like(vel) * (sn.vel_noise + sn.vel_noise_rel * vel.norm(dim=-1, keepdim=True)) * sc
            omega = omega + self.gyro_bias + torch.randn_like(omega) * sn.gyro_noise_obs * sc
        Rt = R.transpose(1, 2)

        def to_body(v: torch.Tensor) -> torch.Tensor:
            return (Rt @ v.unsqueeze(-1)).squeeze(-1)

        head = [to_body(vel) / 10.0, R[:, 2, :], omega / 10.0, self.prev_action]  # R[2] = world up axis in body (no yaw)
        if res is not None:
            head[0] = head[0] + res[:, 6:9] / 10.0
            head[1] = torch.nn.functional.normalize(head[1] + res[:, 9:12], dim=-1)
        last = self._last_slot()
        gate_parts, tokens = [], []
        for j in range(cfg.n_obs_gates):
            idx = torch.minimum(self.gate_idx + j, last)
            if res is not None:  # map-based EKF: every gate is "seen"; gates beyond the second reuse its residual
                c, nrm, _ = self._gate_frame(idx)
                rel_b = to_body(c - q.pos) + res[:, 0:3] if j == 0 else to_body(c - q.pos) + res[:, 3:6]
                one = torch.ones(self.n, 1, device=self.device)
                parts_j = [rel_b / 5.0, to_body(nrm), self.gate_size[:, None]]
                gate_parts += parts_j + ([one] if cfg.vis_obs else [])
            elif cfg.vis_obs:
                seen = self.vis_seen[self.arange, idx]
                if noisy:
                    rel = self.vis_w[self.arange, idx] - (q.pos + self.odo_drift)
                    yw = self.vis_yaw[self.arange, idx]
                    nrm = torch.stack((torch.cos(yw), torch.sin(yw), torch.zeros_like(yw)), -1)
                else:
                    c, nrm, _ = self._gate_frame(idx)
                    rel = c - q.pos
                f_ = seen.float()[:, None]
                m_ = f_ if noisy else torch.ones_like(f_)  # critic keeps privileged truth for unseen gates
                gate_parts += [to_body(rel) / 5.0 * m_, to_body(nrm) * m_, self.gate_size[:, None] * m_, f_]
            else:
                c, nrm, _ = self._gate_frame(idx, noisy)
                rel = c - q.pos
                if sc > 0:
                    rel = rel + torch.randn_like(rel) * (sn.gate_noise + sn.gate_noise_dist * rel.norm(dim=-1, keepdim=True)) * sc
                gate_parts += [to_body(rel) / 5.0, to_body(nrm), self.gate_size[:, None]]
            if cfg.markers:
                tokens.append(self._tokens(idx))

        rel = self.obs_xy - q.pos[:, None, :2]
        dist = rel.norm(dim=-1)
        dist = torch.where(self.obs_on, dist, torch.full_like(dist, 1e6))
        k = min(cfg.n_obs_obstacles, self.obs_xy.shape[1])
        near = dist.topk(k, dim=-1, largest=False).indices
        b = self.arange[:, None]
        rel_k = rel[b, near]
        if sc > 0:
            rel_k = rel_k + torch.randn_like(rel_k) * sn.obstacle_noise * sc
        present = self.obs_on[b, near].float()
        rel3 = torch.cat((rel_k, torch.zeros_like(rel_k[..., :1])), -1) * present[..., None]
        rel3_b = (Rt[:, None] @ rel3.unsqueeze(-1)).squeeze(-1) / 5.0
        obs_feat = torch.cat((rel3_b, (self.obs_r[b, near] * present)[..., None], present[..., None]), -1)
        return head, gate_parts, [obs_feat.reshape(self.n, -1)], tokens

    def teacher_obs(self) -> torch.Tensor:
        """Clean privileged observation in the actor layout (what the teacher is queried with)."""
        head, gates, obst, tokens = self._parts(False)
        parts = head + gates + obst + tokens
        if self.cfg.landing:
            parts.append(self.in_landing().float()[:, None])
        obs = torch.cat(parts, -1)
        if self.cfg.policy_sees_params:
            obs = torch.cat((obs, self.quad.param_obs()), -1)
        if self.cfg.obs_history:
            obs = torch.cat((obs, self.hist[:, -self.cfg.obs_history:].reshape(self.n, -1)), -1)
        if self.cfg.age_obs:
            obs = torch.cat((obs, torch.zeros(self.n, self.cfg.n_obs_gates, device=self.device)), -1)
        return obs

    def compute_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(policy obs with estimator / perception errors, privileged clean critic obs)."""
        cfg, q = self.cfg, self.quad

        def assemble(head, gates, obst, tokens):
            parts = head + gates + obst + tokens
            if cfg.landing:
                parts.append(self.in_landing().float()[:, None])
            return torch.cat(parts, -1)

        clean = assemble(*self._parts(False))
        params = q.param_obs()
        t_frac = (self.steps.float() / self.max_steps.float())[:, None]
        critic = torch.cat((clean, params, t_frac), -1)

        if cfg.sensors.scale > 0:
            head, gates, obst, tokens = self._parts(True)
            gate_vec = torch.cat(gates, -1)
            self._gate_part = gate_vec
            if self.gate_last is not None:  # frozen (dropped-out) gate observation
                gate_vec = torch.where(self.hold[:, None], self.gate_last, gate_vec)
            policy = assemble(head, [gate_vec], obst, tokens)
            self._frame = torch.cat((head[0], head[2], head[3]), -1)
        else:
            policy = clean
            head_c, gates_c = self._parts(False)[:2]
            self._gate_part = torch.cat(gates_c, -1)
            self._frame = torch.cat((head_c[0], head_c[2], head_c[3]), -1)
        if cfg.policy_sees_params:
            policy = torch.cat((policy, params), -1)
        if cfg.obs_history:
            policy = torch.cat((policy, self.hist[:, -cfg.obs_history:].reshape(self.n, -1)), -1)
        if cfg.age_obs:
            last = self._last_slot()
            ages = torch.stack([self.vis_age[self.arange, torch.minimum(self.gate_idx + j, last)] for j in range(cfg.n_obs_gates)], -1)
            policy = torch.cat((policy, ages), -1)
        if self.opp is not None:
            policy = torch.cat((policy, self.opp.obs(True)), -1)
            critic = torch.cat((critic, self.opp.obs(False)), -1)
        return policy, critic

    def imu_obs(self) -> torch.Tensor:
        """IMU readings [N, 6]: gyro [rad/s]/10, accelerometer specific force [m/s^2]/20.

        Includes bias, scale error, noise, thrust-proportional vibration, range saturation
        (accelerometer +-16 g) and inconsistent glitches once the accelerometer is near its limit.
        """
        cfg, q, sn = self.cfg, self.quad, self.cfg.sensors
        sc = sn.scale
        thr = q.thr[:, None]
        gyro = q.omega * self.imu_gyro_scale + self.gyro_bias + torch.randn_like(q.omega) * cfg.gyro_noise
        accel = q.spec_b * self.imu_accel_scale + self.accel_bias + torch.randn_like(q.spec_b) * cfg.accel_noise
        if sc > 0:
            gyro = gyro + torch.randn_like(gyro) * sn.vib_gyro * thr * sc
            accel = accel + torch.randn_like(accel) * sn.vib_accel * thr * sc
            lim = sn.accel_sat_g * GRAVITY
            near = (accel.abs().max(dim=-1, keepdim=True).values > 0.85 * lim).float()
            accel = accel + torch.randn_like(accel) * sn.sat_glitch * near * sc
            accel = accel.clamp(-lim, lim)
            gyro = gyro.clamp(-sn.gyro_sat, sn.gyro_sat)
        return torch.cat((gyro / 10.0, accel / 20.0), -1)

    # ---- step ---------------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        cfg, q = self.cfg, self.quad
        a = action.clamp(-1.0, 1.0)
        sn = cfg.sensors
        p_prev = q.pos.clone()
        eff = a.clone()
        if sn.scale > 0:
            frac = (self.steps.float() / self.max_steps.float()).clamp(0, 1)
            if cfg.vehicle.direct_motor:  # battery sag scales every motor
                eff = 2.0 * (0.5 * (a + 1.0) * (1.0 - self.sag * frac)[:, None]) - 1.0
            else:
                eff[:, 0] = 2.0 * (0.5 * (a[:, 0] + 1.0) * (1.0 - self.sag * frac)) - 1.0  # battery sag
            decay = torch.exp(torch.tensor(-cfg.dt * cfg.decimation / sn.gust_tau, device=self.device))
            self.gust = self.gust * decay + torch.randn_like(self.gust) * sn.gust * sn.scale * torch.sqrt(1 - decay**2)
            q.ext_acc[:] = self.wind + self.gust
        for sub in range(cfg.decimation):
            use_prev = (sub < self.act_delay)[:, None]
            q.step(torch.where(use_prev, self.prev_eff, eff), cfg.dt)
        self.prev_eff = eff
        if cfg.omega_limit > 0:  # direct motor control can spin up without a rate loop: sanitise non-finite states, spin-out = crash
            bad = ~(torch.isfinite(q.pos).all(-1) & torch.isfinite(q.vel).all(-1) & torch.isfinite(q.omega).all(-1) & torch.isfinite(q.quat).all(-1))
            if bad.any():
                q.pos[bad], q.vel[bad], q.omega[bad] = 0.0, 0.0, 0.0
                q.quat[bad] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
            spin_out = bad | (q.omega.norm(dim=-1) > cfg.omega_limit)
        else:
            spin_out = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        p_new = q.pos
        hit_opp = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        if self.opp is not None:
            self.opp.step(cfg.dt * cfg.decimation)
            hit_opp = self.opp.hit()

        in_land = self.in_landing()
        tgt = torch.minimum(self.gate_idx, self._last_slot())
        gc, nrm, right = self._gate_frame(tgt)
        s_prev = ((p_prev - gc) * nrm).sum(-1)
        s_new = ((p_new - gc) * nrm).sum(-1)
        d_prev = (p_prev - gc).norm(dim=-1)
        d_new = (p_new - gc).norm(dim=-1)

        crossing = (s_prev < 0) & (s_new >= 0) & ~in_land
        t = (-s_prev / (s_new - s_prev).clamp_min(1e-9)).clamp(0, 1)[:, None]
        p_c = p_prev + t * (p_new - p_prev)
        lat = ((p_c - gc) * right).sum(-1).abs()
        ver = (p_c[:, 2] - gc[:, 2]).abs()
        half = 0.5 * self.gate_size
        fr = cfg.track.gate_frame
        half_in = (half - q.radius if cfg.gate_body else half) - cfg.gate_shrink  # drone body: its centre must clear the opening by the body radius (+ training shrink)
        if cfg.gate_box:  # geometry-aware: lateral / vertical extent of the body box for the attitude at the crossing (support function of the box)
            Rb_ = quat_to_rot(q.quat)
            hb = torch.stack((q.arm * 0.7071 + cfg.box_prop, q.arm * 0.7071 + cfg.box_prop, torch.full_like(q.arm, cfg.box_hz)), -1)  # [n,3] half extents x,y,z
            ext_lat = ((right[:, :, None] * Rb_).sum(1).abs() * hb).sum(-1)
            ext_ver = (Rb_[:, 2, :].abs() * hb).sum(-1)
            inside = (lat < half - ext_lat - cfg.gate_shrink) & (ver < half - ext_ver - cfg.gate_shrink)
        else:
            inside = (lat < half_in) & (ver < half_in)
        frame = (lat < half + fr) & (ver < half + fr)
        passed = crossing & inside
        hit_frame = crossing & frame & ~inside

        hit_decoy = torch.zeros_like(passed)
        if cfg.markers:
            dc = self.decoy_pos[self.arange, tgt]
            lat_d = ((p_c - dc) * right).sum(-1).abs()
            ver_d = (p_c[:, 2] - dc[:, 2]).abs()
            hit_decoy = crossing & self.decoy_on[self.arange, tgt] & (lat_d < half + fr) & (ver_d < half + fr)
            passed = passed & ~hit_decoy

        # poles: closest point of the xy-segment to each axis
        seg = p_new[:, :2] - p_prev[:, :2]
        seg2 = (seg * seg).sum(-1, keepdim=True).clamp_min(1e-9)
        tt = (((self.obs_xy - p_prev[:, None, :2]) * seg[:, None]).sum(-1) / seg2).clamp(0, 1)
        closest = p_prev[:, None, :2] + tt[..., None] * seg[:, None]
        dmin = (closest - self.obs_xy).norm(dim=-1)
        z_ok = p_new[:, 2] < cfg.track.obs_height
        hit_pole = ((dmin < self.obs_r + q.radius[:, None]) & self.obs_on).any(-1) & z_ok

        hit_ground = p_new[:, 2] < 0.0
        oob = (p_new[:, :2] - self.gate_pos[:, 0, :2]).norm(dim=-1) > cfg.out_of_bounds
        oob |= spin_out
        oob |= ((p_new[:, :2] - self.hall[:, :2]).abs() > self.hall[:, 2:]).any(-1)  # hall walls (nets)
        speed = q.vel.norm(dim=-1)
        if cfg.landing:
            xy_d = (p_new[:, :2] - gc[:, :2]).norm(dim=-1)
            landed = in_land & (xy_d < cfg.land_radius) & (p_new[:, 2] < cfg.land_height) & (speed < cfg.land_speed)
        else:
            landed = torch.zeros_like(passed)
        soft_hit = hit_frame & self.soft
        crashed = ((hit_frame & ~self.soft) | hit_pole | hit_decoy | hit_ground | oob | hit_opp) & ~landed

        self.gate_idx += ((passed | soft_hit) & ~crashed).long()
        self._update_vis()
        finished = landed if cfg.landing else (self.gate_idx >= self.n_gates)
        self.steps += 1
        timeout = (self.steps >= self.max_steps) & ~crashed & ~finished

        rw = cfg.reward
        prog = d_prev - d_new
        if rw.v_max > 0:
            prog = prog.clamp(max=rw.v_max * cfg.dt * cfg.decimation)
        off = torch.maximum(lat, ver) / half.clamp_min(1e-6)
        perc = torch.zeros_like(prog)
        if rw.perc > 0:
            Rm = quat_to_rot(q.quat)
            tilt = cfg.cam_tilt
            cam = Rm[:, :, 0] * math.cos(tilt) + Rm[:, :, 2] * math.sin(tilt)  # camera optical axis, world
            to_g = gc - p_new
            cosang = (cam * to_g).sum(-1) / to_g.norm(dim=-1).clamp_min(1e-6)
            ang = torch.acos(cosang.clamp(-1.0, 1.0))
            perc = torch.exp(-(ang**4)) * (~in_land).float()
        pen = torch.zeros_like(prog)
        if rw.perc_pen > 0:
            Rm2 = quat_to_rot(q.quat)
            cam2 = Rm2[:, :, 0] * math.cos(cfg.cam_tilt) + Rm2[:, :, 2] * math.sin(cfg.cam_tilt)
            to_g2 = gc - p_new
            ang2 = torch.acos(((cam2 * to_g2).sum(-1) / to_g2.norm(dim=-1).clamp_min(1e-6)).clamp(-1.0, 1.0))
            pen = (ang2 - math.pi / 3).clamp_min(0.0) * (~in_land).float()
        opp_prox = torch.zeros_like(prog)
        if self.opp is not None and rw.opp_near > 0:
            gap = self.opp.dist() - q.radius[:, None] - cfg.opp_radius
            opp_prox = ((1.0 - gap / cfg.opp_safe).clamp(0.0, 1.0) ** 2).amax(-1)
        near_land = (1.0 - d_new / 4.0).clamp(0.0, 1.0) * in_land.float()
        rew = (
            rw.progress * prog
            + rw.perc * perc
            - rw.perc_pen * pen
            - rw.speed_pen * (speed - rw.speed_cap).clamp_min(0.0)
            - rw.slow_pen * (rw.slow_min - speed).clamp_min(0.0) * (self.steps.float() * (cfg.dt * cfg.decimation) > rw.slow_grace).float()
            + rw.center * passed.float() * (1.0 - off).clamp(0.0, 1.0)
            - rw.align * passed.float() * ((torch.rad2deg(torch.acos(((q.vel * nrm).sum(-1).abs() / speed.clamp_min(1e-3)).clamp(max=1.0))) - rw.align_free).clamp_min(0.0) / 45.0)
            + rw.gate * passed.float()
            + rw.gate_speed * passed.float() * (speed.clamp(max=rw.gate_speed_cap) / 10.0)
            + rw.finish * finished.float()
            + rw.finish_time * finished.float() * (1.0 - self.steps.float() / self.max_steps.float()).clamp_min(0.0)
            - rw.crash * crashed.float()
            - rw.opp_near * opp_prox
            - rw.soft_frame * soft_hit.float()
            - rw.decoy * hit_decoy.float()
            - rw.land_slow * near_land * speed**2
            - rw.rate * (q.omega**2).sum(-1)
            - rw.action_diff * ((a - self.prev_action) ** 2).sum(-1)
            - rw.yaw_cmd * a[:, 3] ** 2
            - rw.yaw_diff * (a[:, 3] - self.prev_action[:, 3]) ** 2
            + rw.alive
        )
        self.prev_action = a
        self.ep_return += rew

        terminated = crashed | finished
        done = terminated | timeout
        if self.rb_res is not None:
            self._rb_advance()
        obs_pre, critic_pre = self.compute_obs()
        info = {
            "final_critic_obs": critic_pre,
            "timeout": timeout,
            "crashed": crashed,
            "finished": finished,
            "decoy": hit_decoy,
            "opp_hit": hit_opp,
            "gates_passed": self.gate_idx.clone().clamp(max=self.n_gates),
            "n_gates": self.n_gates.clone(),
            "ep_return": self.ep_return.clone(),
            "ep_len": self.steps.clone(),
            "twr": q.twr.clone(),
            "track_len": self.track_len.clone(),
            "speed": speed,
            "cause": (hit_frame.long() * 1 + (hit_pole & ~hit_frame).long() * 2 + (hit_ground & ~hit_frame & ~hit_pole).long() * 4
                      + (oob & ~hit_frame & ~hit_pole & ~hit_ground).long() * 5
                      + (hit_opp & ~hit_frame & ~hit_pole & ~hit_ground & ~oob).long() * 6) * crashed.long(),  # 1 frame, 2 pole, 4 ground, 5 out of bounds, 6 opponent
        }
        ids = done.nonzero().squeeze(-1)
        if len(ids) > 0:
            self.reset_idx(ids)
            self._update_vis(drift=False)
            obs_pre, critic_pre = self.compute_obs()
        if cfg.obs_history:
            self.hist = torch.roll(self.hist, -1, dims=1)
            self.hist[:, -1] = self._frame
        if sn.scale > 0:
            # remember the freshly seen gate block, then draw next step's dropout state
            self.gate_last = torch.where(self.hold[:, None], self.gate_last, self._gate_part)
            if len(ids) > 0:
                self.gate_last[ids] = self._gate_part[ids]
            keep = torch.rand(self.n, generator=self.gen, device=self.device)
            self.hold = torch.where(self.hold, keep < sn.gate_dropout_hold, keep < sn.gate_dropout * sn.scale)
            if len(ids) > 0:
                self.hold[ids] = False
            # perception latency: serve an older observation
            self.obs_buf = torch.roll(self.obs_buf, 1, dims=1)
            self.obs_buf[:, 0] = obs_pre
            if len(ids) > 0:
                self.obs_buf[ids] = obs_pre[ids][:, None]
            obs_pre = self.obs_buf[self.arange, self.obs_delay]
        return obs_pre, critic_pre, rew, terminated, done, info
