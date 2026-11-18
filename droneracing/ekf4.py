"""EKF3 with corner-count dependent PnP noise (MonoRace: sigma^2 ~ d^2 / Nc^2).

``GateEKF4.update`` is the loosely coupled whole-gate update of ``GateEKF.update`` with the aided-PnP variance multiplied by
``(4 / Nc) ** nc_pow`` where Nc is the number of detected gate corners inside the image (clamped to 2..4): a gate that is partly out of
view gives a less certain position. Everything else (corner peaks of EKF3, held gate offsets) is unchanged; base classes are untouched.
"""
from __future__ import annotations

import torch

from .ekf import skew
from .ekf3 import GateEKF3
from .estimator import DET_DIM, K_DET


class GateEKF4(GateEKF3):
    def __init__(self, *a, nc_pow: float = 2.0, **kw):
        super().__init__(*a, **kw)
        self.nc_pow = nc_pow

    @torch.no_grad()
    def _base_update(self, feat: torch.Tensor, gate_idx: torch.Tensor, map_c: torch.Tensor, map_n: torch.Tensor, last_idx: torch.Tensor) -> None:
        """feat [n, K*DET_DIM]; map_c / map_n [n, J, 3] map gate centres / normals (map frame) for gate_idx + 0..J-1."""
        n, J = self.n, map_c.shape[1]
        fd = feat.reshape(n, K_DET, DET_DIM)
        ar = torch.arange(n, device=self.dev)
        self.accepted.zero_()
        for s in range(K_DET):
            valid = fd[:, s, 0] > 0
            if not valid.any():
                continue
            z = fd[:, s, 1:4] * 5.0
            cpx = (fd[:, s, 8:16].reshape(n, 4, 2) + 1.0) * 0.5 * torch.tensor([self.img_w, self.img_h], device=self.dev)
            nc = ((cpx[..., 0] >= 0) & (cpx[..., 0] < self.img_w) & (cpx[..., 1] >= 0) & (cpx[..., 1] < self.img_h)).sum(-1).clamp(2, 4).float()
            infl = ((4.0 / nc) ** self.nc_pow)[:, None, None, None]  # [n,1,1,1]
            zn = fd[:, s, 4:7]
            dist = z.norm(dim=-1)
            R = self.rot()
            if self.aided:  # attitude-aided PnP per candidate gate (filter attitude + map gate yaw), as in MonoRace
                z_j = self._aided_positions(fd[:, s, 8:16].reshape(n, 4, 2), R, map_n)  # [n,J,3]
            rel_w = map_c - self.p[:, None, :]  # [n,J,3]; nominal map only: the absolute anchor of the pose
            zhat = (R.transpose(1, 2)[:, None] @ rel_w[..., None])[..., 0]  # [n,J,3]
            if self.aided:
                sig2 = ((self.sig_aided[0] + self.sig_aided[1] * z_j.norm(dim=-1)) ** 2 * infl[..., 0, 0] + self.sig_map ** 2)[..., None, None]  # [n,J,1,1]
            else:
                sig2 = ((self.sig_pnp[0] + self.sig_pnp[1] * dist) ** 2 + self.sig_map ** 2)[:, None, None, None]
            Hj = torch.zeros(n, J, 3, 15, device=self.dev)
            Hj[..., 0:3] = -R.transpose(1, 2)[:, None]
            Hj[..., 6:9] = skew(zhat)
            PHt = self.P[:, None] @ Hj.transpose(2, 3)  # [n,J,15,3]
            S = Hj @ PHt + sig2 * torch.eye(3, device=self.dev)
            y = (z_j if self.aided else z[:, None, :]) - zhat
            m2 = (y[..., None, :] @ torch.linalg.solve(S, y[..., None]))[..., 0, 0]  # [n,J]
            best = m2.argmin(-1)
            ok = valid & (m2[ar, best] < self.chi2) & ((gate_idx + best) <= last_idx)
            if not ok.any():
                continue
            Hb, yb = Hj[ar, best], y[ar, best]
            if self.aided:
                z = z_j[ar, best]
            sb = sig2.expand(n, J, 1, 1)[ar, best] if self.aided else sig2[:, 0]
            self._apply(Hb, yb, sb * torch.eye(3, device=self.dev), ok)
            self.accepted += ok.float()
            # gate normal (attitude cue), loosely weighted and gated
            R = self.rot()
            nhat = (R.transpose(1, 2) @ map_n[ar, best][..., None])[..., 0]
            yn = zn - nhat
            okn = ok & (zn.norm(dim=-1) > 0.5) & (yn.norm(dim=-1) < 0.8)
            Hn = torch.zeros(n, 3, 15, device=self.dev)
            Hn[:, :, 6:9] = skew(nhat)
            self._apply(Hn, yn, (self.sig_nrm ** 2) * torch.eye(3, device=self.dev).expand(n, 3, 3), okn)
            # refine the held gate position from the (updated) measurement
            R = self.rot()
            meas_w = self.p + (R @ z[..., None])[..., 0]
            gi = (gate_idx + best).clamp(max=self.dg.shape[1] - 1)
            cur = self.dg[ar, gi]
            new = (1 - self.kappa) * cur + self.kappa * (meas_w - map_c[ar, best])
            self.dg[ar[ok], gi[ok]] = new[ok].clamp(-1.5, 1.5)


    @torch.no_grad()
    def update(self, feat, gate_idx, map_c, map_n, last_idx, peaks=None):
        # EKF3.update calls GateEKF.update(self, ...) for the whole-gate part; route that part through the corner-count version
        import droneracing.ekf3 as _m
        orig = _m.GateEKF.update
        _m.GateEKF.update = lambda s, f, g, c, nn, l: GateEKF4._base_update(s, f, g, c, nn, l)
        try:
            super().update(feat, gate_idx, map_c, map_n, last_idx, peaks=peaks)
        finally:
            _m.GateEKF.update = orig
