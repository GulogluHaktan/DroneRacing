"""GateEKF7: GateEKF5 with a tunable random walk of the accelerometer bias (and optionally the gyro bias).

The IMU of the simulator (and the real one) has a scale error of about 1 % per axis. At a specific force of 10-40 m/s^2 it looks like a bias of
0.1-0.4 m/s^2 that changes with the thrust. The base filter has a nearly constant bias (random walk 2e-3 m/s^2/sqrt(s)), so the velocity error
grows by 0.1-0.4 m/s per second between the gate fixes. A larger random walk lets the bias state follow the scale error.
Base classes are unchanged. ``ba_walk`` = 0 gives GateEKF5.
"""
from __future__ import annotations

import torch

from .ekf5 import GateEKF5


class GateEKF7(GateEKF5):
    def __init__(self, *a, ba_walk: float = 0.0, bg_walk: float = 0.0, lap_len: int = 0, **kw):
        super().__init__(*a, **kw)
        self.ba_walk, self.bg_walk, self.lap_len = ba_walk, bg_walk, lap_len
        self.lap_copied = torch.zeros(self.n, dtype=torch.bool, device=self.dev)

    def reset(self, ids, p, v, q):
        super().reset(ids, p, v, q)
        self.lap_copied[ids] = False

    @torch.no_grad()
    def update(self, feat, gate_idx, map_c, map_n, last_idx, peaks=None, delay: int = 0):
        """Two-lap gate memory: gate slot k + lap_len is the same physical gate as slot k. When the first gate of lap 2 comes up, the corrections
        held for lap 1 become the start values of lap 2."""
        L = self.lap_len
        if L > 0 and self.dg.shape[1] >= 2 * L:
            new = (gate_idx >= L) & ~self.lap_copied
            if bool(new.any()):
                ids = torch.nonzero(new).flatten()
                self.dg[ids, L : 2 * L] = self.dg[ids, 0:L]
                self.lap_copied[ids] = True
        return super().update(feat, gate_idx, map_c, map_n, last_idx, peaks=peaks, delay=delay)

    @torch.no_grad()
    def predict(self, imu, dt):
        super().predict(imu, dt)
        if self.ba_walk > 0:
            self.P[:, 9:12, 9:12] += torch.eye(3, device=self.dev) * (self.ba_walk * dt) ** 2
        if self.bg_walk > 0:
            self.P[:, 12:15, 12:15] += torch.eye(3, device=self.dev) * (self.bg_walk * dt) ** 2
        if self.hist:
            self.hist[-1]["P"] = self.P.clone()
