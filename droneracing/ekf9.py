"""GateEKF9: drag-aided velocity (quadrotor accelerometer model) on top of the delay compensation of GateEKF5.

A quadrotor's thrust acts along the body z axis only, so the body x / y specific force that the accelerometer measures is the
aerodynamic drag: a_b,xy = -(c |v| + c_lin) v_b,xy (+ accelerometer bias). This makes the horizontal velocity observable from
the IMU alone and limits the drift of the velocity between gate detections (Leishman et al. 2014; MonoRace replaces the
accelerometer by the drone model when it saturates). The drag coefficients come from a calibration (bench / flight log): ``set_drag``.
"""
from __future__ import annotations

import torch

from .ekf import skew
from .ekf5 import GateEKF5


class GateEKF9(GateEKF5):
    def __init__(self, *a, sig_drag: float = 0.5, drag_model_err: float = 0.15, drag_every: int = 1, **kw):
        super().__init__(*a, **kw)
        self.sig_drag, self.drag_model_err, self.drag_every = sig_drag, drag_model_err, drag_every
        self.c_quad = torch.zeros(self.n, device=self.dev)  # [1/m] quadratic drag / mass
        self.c_lin = torch.zeros(self.n, device=self.dev)  # [1/s] linear drag
        self._k = 0
        self.drag_used = torch.zeros(self.n, device=self.dev)

    def set_drag(self, c_quad: torch.Tensor, c_lin: torch.Tensor) -> None:
        self.c_quad, self.c_lin = c_quad.to(self.dev).float(), c_lin.to(self.dev).float()

    @torch.no_grad()
    def predict(self, imu, dt):
        super(GateEKF5, self).predict(imu, dt)  # IMU propagation of the base filter (GateEKF5.predict would store the snapshot too early)
        self._k += 1
        if self._k % self.drag_every == 0 and (self.c_quad > 0).any():
            self._drag_update(imu)
        self.hist.append({k: getattr(self, k).clone() for k in ("p", "v", "q", "ba", "bg", "P")})

    def _drag_update(self, imu):
        n = self.n
        f_raw = imu[:, 3:] * 20.0
        R = self.rot()
        Rt = R.transpose(1, 2)
        v_b = (Rt @ self.v[..., None])[..., 0]
        s = self.v.norm(dim=-1).clamp_min(0.3)
        k = self.c_quad * s + self.c_lin  # [n]
        a_pred = -k[:, None] * v_b + self.ba  # [n,3]
        y = (f_raw - a_pred)[:, :2]
        # Jacobians (rows x, y of the body specific force)
        Jv = -(k[:, None, None] * Rt + self.c_quad[:, None, None] * (v_b[..., None] @ self.v[:, None, :]) / s[:, None, None])
        Jth = -k[:, None, None] * skew(v_b)
        H = torch.zeros(n, 2, 15, device=self.dev)
        H[:, :, 3:6] = Jv[:, :2]
        H[:, :, 6:9] = Jth[:, :2]
        H[:, 0, 9] = 1.0
        H[:, 1, 10] = 1.0
        sig2 = self.sig_drag ** 2 + (self.drag_model_err * (k * v_b[:, :2].norm(dim=-1))) ** 2
        Rm = torch.diag_embed(sig2[:, None].expand(n, 2))
        sat = f_raw.abs().amax(-1) > 0.85 * 16 * 9.81
        S = H @ self.P @ H.transpose(1, 2) + Rm
        d2 = (y[:, None, :] @ torch.linalg.solve(S, y[..., None]))[:, 0, 0]
        ok = (~sat) & (d2 < 13.8) & (self.c_quad > 0)  # chi2 2 dof, 99.9 %
        self._apply(H, y, Rm, ok)
        self.drag_used = ok.float()
