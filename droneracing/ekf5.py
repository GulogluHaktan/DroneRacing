"""GateEKF5: GateEKF3 with image-delay compensation (MonoRace: gate position error 0.289 -> 0.103 m).

A detection that reaches the filter ``delay`` policy steps after the image was taken describes the world at the capture time.
At 15 m/s and 50 ms that is 0.75 m, and the attitude moves by up to 0.5 rad. Method (state roll-back, linear correction):
1. After every ``predict`` the filter stores (p, v, q, ba, bg, P).
2. ``update(..., delay=d)`` loads the snapshot of d steps ago, runs the unchanged GateEKF3 update on it (pose, covariance and
   gate PnP all at capture time) and measures the correction (dp, dv, dtheta, dba, dbg, dP).
3. The same correction is added to the CURRENT state (position and velocity errors at capture time carry over to now to first
   order, the attitude correction is rotated into the world frame). Updates applied between capture time and now are kept.
The gate offsets ``dg`` use the capture-time pose, so the held gate position is not biased by the vehicle motion.
Base classes are unchanged. delay=0 gives GateEKF3.
"""
from __future__ import annotations

from collections import deque

import torch

from .ekf3 import GateEKF3
from .quad import quat_exp, quat_mul, quat_to_rot


class GateEKF5(GateEKF3):
    def __init__(self, *a, max_delay: int = 12, **kw):
        super().__init__(*a, **kw)
        self.hist: deque = deque(maxlen=max_delay + 1)

    def reset(self, ids, p, v, q):
        super().reset(ids, p, v, q)
        for h in self.hist:  # old snapshots of the reset envs are from another episode
            h["p"][ids], h["v"][ids], h["q"][ids] = self.p[ids], self.v[ids], self.q[ids]
            h["ba"][ids], h["bg"][ids], h["P"][ids] = self.ba[ids], self.bg[ids], self.P[ids]

    @torch.no_grad()
    def predict(self, imu, dt):
        super().predict(imu, dt)
        self.hist.append({k: getattr(self, k).clone() for k in ("p", "v", "q", "ba", "bg", "P")})

    def _pre_update(self, feat, gate_idx, map_c, map_n, peaks):
        """Hook for subclasses (GateEKF6): called with the state at the capture time before the update; returns (feat, peaks)."""
        return feat, peaks

    @torch.no_grad()
    def update(self, feat, gate_idx, map_c, map_n, last_idx, peaks=None, delay: int = 0):
        if delay <= 0 or len(self.hist) <= delay:
            feat, peaks = self._pre_update(feat, gate_idx, map_c, map_n, peaks)
            return super().update(feat, gate_idx, map_c, map_n, last_idx, peaks=peaks)
        snap = self.hist[-1 - delay]  # (state at the capture time)
        now = {k: getattr(self, k) for k in ("p", "v", "q", "ba", "bg", "P")}
        for k in now:  # roll back to the capture time
            setattr(self, k, snap[k].clone())
        feat, peaks = self._pre_update(feat, gate_idx, map_c, map_n, peaks)
        super().update(feat, gate_idx, map_c, map_n, last_idx, peaks=peaks)
        dp, dv = self.p - snap["p"], self.v - snap["v"]
        dba, dbg = self.ba - snap["ba"], self.bg - snap["bg"]
        Rc = quat_to_rot(snap["q"])
        qrel = quat_mul(_conj(snap["q"]), self.q)  # right perturbation in the capture-time body frame
        dth = _log(qrel)
        dP = self.P - snap["P"]
        # corrected current state
        self.p, self.v = now["p"] + dp, now["v"] + dv
        self.q = quat_mul(quat_exp((Rc @ dth[..., None])[..., 0]), now["q"])
        self.q = self.q / self.q.norm(dim=-1, keepdim=True)
        self.ba, self.bg = now["ba"] + dba, now["bg"] + dbg
        P = now["P"] + dP
        P = 0.5 * (P + P.transpose(1, 2))
        bad = ~torch.isfinite(P).all(-1).all(-1)
        P = torch.where(bad[:, None, None], now["P"], P)
        lam, V = torch.linalg.eigh(P)  # the reduction dP can leave P indefinite: clamp the eigenvalues
        self.P = (V * lam.clamp_min(1e-10)[:, None, :]) @ V.transpose(1, 2)
        self.hist[-1] = {k: getattr(self, k).clone() for k in ("p", "v", "q", "ba", "bg", "P")}


def _conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def _log(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (w, x, y, z) -> rotation vector."""
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return v / s * 2.0 * torch.atan2(s, q[..., :1])
