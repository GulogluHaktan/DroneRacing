"""Estimator-in-the-loop training environment: the actor observes the output of the real gate filter (GateEKF5, delay compensation),
fed by an analytic camera + detector model, instead of the true state with parametric noise.

The policy was trained on near-true observations and flies racing lines that have no margin for the structured errors of the camera
line (held gate estimates out of view, 50 ms delay, correlated errors near the gate). Privileged-learning and Swift-style work train
the deployed policy on the observations of the deployed estimator; this wrapper does that inside the fast torch simulator.
Observation layout = isaac/eval_vision.py (EKF branch): [vel/10, up, gyro/10, prev action, per gate (rel/5, normal, size, flag)] + history.
The critic keeps the privileged clean observation of the wrapped env.
"""
from __future__ import annotations

import collections

import torch

from .detector_model import sim_detections
from .ekf5 import GateEKF5
from .estimator import det_features
from .extrinsic import distort_feat, rot_xyz
from .quad import quat_exp, quat_mul

# perception domain randomization (per drone, redrawn every episode): what the real camera line will do worse than the simulator.
# recall / noise of the detector, true image delay [policy steps] while the filter assumes cam_delay, residual mount rotation after
# calibration [deg per axis], extra share of lost frames (radio interference). "bad_day" = fixed pessimistic values for evaluation.
DR_TRAIN = dict(recall=(0.85, 0.97), noise=(1.0, 1.6), delay=(3, 7), mount=(0.0, 0.7), lost=(0.0, 0.15))
DR_BAD_DAY = dict(recall=(0.85, 0.85), noise=(1.5, 1.5), delay=(3, 7), mount=(0.7, 0.7), lost=(0.15, 0.15))


class EkfLoopEnv:
    def __init__(self, env, H: int = 308, W: int = 410, hfov: float = 128.0, cam_hz: float = 30.0, cam_delay: int = 5,
                 recall: float = 0.95, noise_scale: float = 1.0, ekf_kw: dict | None = None, dr: dict | None = None, calib: str = ""):
        self.env = env
        cfg = env.cfg
        self.n, self.dev = env.n, env.device
        self.H, self.W, self.hfov = H, W, hfov
        self.recall, self.noise_scale = recall, noise_scale
        self.dt = cfg.dt * cfg.decimation
        self.per = max(1, round(1.0 / self.dt / cam_hz))
        self.cam_delay = cam_delay
        kw = dict(sig_map=0.25, kappa=0.3, kappa_corner=0.15, aided_perm=True, n_map=32)
        kw.update(ekf_kw or {})
        self.ekf = GateEKF5(self.n, self.dev, H=H, W=W, hfov=hfov, cam_tilt=cfg.cam_tilt, **kw)
        self.K = cfg.obs_history
        self.J = cfg.n_obs_gates
        assert not cfg.age_obs and not cfg.policy_sees_params, "layout of eval_vision.py (EKF branch) only"
        self.hist = torch.zeros(self.n, self.K, 10, device=self.dev)
        self.empty = det_features([torch.zeros(0, 9)] * self.n, H, W, self.dev, cam_tilt=cfg.cam_tilt, hfov_deg=hfov)
        self.dr = dr
        self.calib = torch.load(calib, map_location="cpu") if calib else None  # scripts/fit_detector.py table (real detector in Isaac)
        self.max_delay = max(cam_delay, int(dr["delay"][1]) if dr else cam_delay)
        self.dq = collections.deque(maxlen=self.max_delay + 1)
        n = self.n
        self.rec = torch.full((n,), recall, device=self.dev)
        self.nsc = torch.full((n,), noise_scale, device=self.dev)
        self.dly = torch.full((n,), cam_delay, dtype=torch.long, device=self.dev)
        self.E = torch.eye(3, device=self.dev).repeat(n, 1, 1)
        self.lost = torch.zeros(n, device=self.dev)
        self.step_i = 0
        self.policy_dim = env.policy_dim
        self.critic_dim = env.critic_dim

    def __getattr__(self, k):  # everything else from the wrapped env (device, cfg, quad, ...)
        return getattr(self.env, k)

    def _map_gates(self):
        e = self.env
        last = e._last_slot()
        idx = [torch.minimum(e.gate_idx + j, last) for j in range(self.J)]
        cs, ns = zip(*[e._gate_frame(i, noisy=True)[:2] for i in idx])
        return torch.stack(cs, 1), torch.stack(ns, 1), last

    def _draw(self, ids):
        if not self.dr:
            return
        k, d = len(ids), self.dev
        u = lambda lo_hi: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(k, device=d)  # noqa: E731
        self.rec[ids], self.nsc[ids], self.lost[ids] = u(self.dr["recall"]), u(self.dr["noise"]), u(self.dr["lost"])
        self.dly[ids] = torch.randint(int(self.dr["delay"][0]), int(self.dr["delay"][1]) + 1, (k,), device=d)
        m = u(self.dr["mount"])[:, None]
        self.E[ids] = rot_xyz((torch.rand(k, 3, device=d) * 2 - 1) * m)

    def _ekf_reset(self, ids):
        self._draw(ids)
        e = self.env
        k = len(ids)
        q = quat_mul(e.quad.quat[ids], quat_exp(torch.randn(k, 3, device=self.dev) * 0.02))
        self.ekf.reset(ids, e.quad.pos[ids] + torch.randn(k, 3, device=self.dev) * 0.05, e.quad.vel[ids], q)
        self.hist[ids] = 0.0
        for i in range(len(self.dq)):  # queued detections belong to the previous episode
            self.dq[i] = self.dq[i].clone()
            self.dq[i][ids] = self.empty[ids]

    @torch.no_grad()
    def _policy_obs(self, imu, predict: bool = True):
        e = self.env
        if self.step_i % self.per == 0:
            if self.dr:
                feat = sim_detections(e, None, self.H, self.W, recall=self.rec, noise_scale=self.nsc, hfov_deg=self.hfov, calib=self.calib)
                feat = distort_feat(feat, self.E, self.H, self.W, self.hfov, e.cfg.cam_tilt)  # residual mount rotation
                gone = torch.rand(self.n, device=self.dev) < self.lost
                feat = torch.where(gone[:, None], self.empty, feat)
            else:
                feat = sim_detections(e, None, self.H, self.W, recall=self.recall, noise_scale=self.noise_scale, hfov_deg=self.hfov, calib=self.calib)
        else:
            feat = self.empty
        self.dq.append(feat)
        L = len(self.dq)
        if self.dr:  # true delay per drone (the filter still assumes cam_delay)
            Q = torch.stack(list(self.dq))  # [L, n, F]
            pos = (L - 1 - self.dly).clamp_min(-1)
            feat_d = torch.where((pos >= 0)[:, None], Q[pos.clamp_min(0), torch.arange(self.n, device=self.dev)], self.empty)
        else:
            feat_d = self.dq[-1 - self.cam_delay] if L > self.cam_delay else self.empty
        mc, mn, last = self._map_gates()
        if predict:
            self.ekf.predict(imu, self.dt)
        self.ekf.update(feat_d, e.gate_idx, mc, mn, last, peaks=None, delay=self.cam_delay)
        vb, up, rel, nrm = self.ekf.outputs(e.gate_idx, last, mc, mn, n_out=self.J)
        vel, rel = vb / 10.0, rel / 5.0
        one = torch.ones(self.n, 1, device=self.dev)
        parts = [vel, up, imu[:, :3], e.prev_action]
        for j in range(self.J):
            parts += [rel[:, j], nrm[:, j], e.gate_size[:, None], one]
        obs = torch.cat(parts + [self.hist.reshape(self.n, -1)], -1)
        if self.env.opp is not None:  # opponent block (nearest drones in the camera view) at the end, as in RaceEnv
            obs = torch.cat((obs, self.env.opp.obs(True)), -1)
        self.hist = torch.roll(self.hist, -1, dims=1)
        self.hist[:, -1] = torch.cat((vel, imu[:, :3], e.prev_action), -1)
        self.step_i += 1
        return obs

    @torch.no_grad()
    def compute_obs(self):
        _, cobs = self.env.compute_obs()
        self._ekf_reset(torch.arange(self.n, device=self.dev))
        obs = self._policy_obs(self.env.imu_obs(), predict=False)
        assert obs.shape[1] == self.policy_dim, (obs.shape, self.policy_dim)
        return obs, cobs

    @torch.no_grad()
    def step(self, act):
        _, cobs, rew, term, done, info = self.env.step(act)
        if done.any():
            self._ekf_reset(torch.nonzero(done).flatten())
        obs = self._policy_obs(self.env.imu_obs())
        return obs, cobs, rew, term, done, info
