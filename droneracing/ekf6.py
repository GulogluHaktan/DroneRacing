"""GateEKF6: GateEKF5 with an online estimate of the camera mount rotation error (3 angles, camera frame).

A mount error of a few degrees is unknown to the filter and shifts every gate corner by the same rotation in the camera frame (3 deg at
f = 100 px is 5 px; PnP then puts the gate 0.3-1 m off at 10 m). MonoRace refines the extrinsics from the gate masks offline. Here the
correction runs online: before every update (state rolled back to the capture time by GateEKF5) the detected corners are rotated
by the current estimate ``e``; the pre-update corner residual against the projected map gate is solved for a small rotation
(3 unknowns from 4 corners, least squares) and added to ``e`` with a small gain, so that the average residual goes to zero over many
gates. Pixel shift of a small camera rotation e: (du, dv) = f * (e_y - e_z * y, -e_x + e_z * x), (x, y) = normalized image coordinates.
Applies to whole-gate detections (feat) and to corner peaks. Base classes are unchanged. ``ext_gain`` 0 = off (GateEKF5).
"""
from __future__ import annotations

import math

import torch

from .ekf5 import GateEKF5
from .estimator import DET_DIM, K_DET, exp_so3


class GateEKF6(GateEKF5):
    def __init__(self, *a, ext_gain: float = 0.02, ext_max_deg: float = 6.0, ext_gate_px: float = 40.0, ext_min_ext: float = 25.0, **kw):
        super().__init__(*a, **kw)
        self.ext_gain, self.ext_max, self.ext_gate_px, self.ext_min_ext = ext_gain, math.radians(ext_max_deg), ext_gate_px, ext_min_ext
        self.e = torch.zeros(self.n, 3, device=self.dev)

    def reset(self, ids, p, v, q):
        super().reset(ids, p, v, q)
        # the mount error is a property of the airframe, not of the episode: keep self.e on a reset

    def _rotate_px(self, uv: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """uv [n,...,2] pixels; rotate the viewing rays by exp(e) (e [n,3]) and project back."""
        n = uv.shape[0]
        sh = uv.shape
        x = (uv[..., 0] - self.img_w / 2) / self.f
        y = (uv[..., 1] - self.img_h / 2) / self.f
        r = torch.stack((x, y, torch.ones_like(x)), -1).reshape(n, -1, 3)
        r2 = r @ exp_so3(e).transpose(1, 2)  # row vectors: (R r)^T = r^T R^T
        z = r2[..., 2].clamp_min(1e-3)
        out = torch.stack((self.f * r2[..., 0] / z + self.img_w / 2, self.f * r2[..., 1] / z + self.img_h / 2), -1)
        return out.reshape(sh)

    @torch.no_grad()
    def _pre_update(self, feat, gate_idx, map_c, map_n, peaks):
        """Called by GateEKF5.update with the state at the capture time. Returns the corrected (feat, peaks)."""
        n = self.n
        wh = torch.tensor([self.img_w, self.img_h], device=self.dev)
        fd = feat.reshape(n, K_DET, DET_DIM).clone()
        corners = (fd[:, :, 8:16].reshape(n, K_DET, 4, 2) + 1.0) * 0.5 * wh  # px
        corners = self._rotate_px(corners, self.e)
        # innovation against the projected map gates (J candidates): nearest centroid, best of 8 corner orders
        J = map_c.shape[1]
        cw = self._corners_world(map_c + self.dg_for(gate_idx, gate_idx + J, J), map_n)
        uv_p, _, Xc = self._project(cw, self.rot())  # [n,J,4,2]
        ok_p = (Xc[..., 2] > 0.3).all(-1)  # [n,J]
        valid = fd[:, :, 0] > 0
        cen_d = corners.mean(2)  # [n,K,2]
        cen_p = uv_p.mean(2)  # [n,J,2]
        dd = (cen_d[:, :, None] - cen_p[:, None]).norm(dim=-1)  # [n,K,J]
        dd = torch.where(ok_p[:, None], dd, torch.full_like(dd, 1e6))
        best = dd.argmin(-1)  # [n,K]
        pred = uv_p[torch.arange(n, device=self.dev)[:, None], best]  # [n,K,4,2]
        ext = (corners.amax(2) - corners.amin(2)).amax(-1)  # [n,K]
        errs, perms = [], []
        for k in range(8):
            c = torch.roll(corners if k < 4 else corners[:, :, [1, 0, 3, 2]], k % 4, dims=2)
            errs.append((c - pred).norm(dim=-1).mean(-1)); perms.append(c)
        E = torch.stack(errs, 0)
        kb = E.argmin(0)
        cbest = torch.stack(perms, 0)[kb, torch.arange(n, device=self.dev)[:, None], torch.arange(K_DET, device=self.dev)[None]]
        res = pred - cbest  # [n,K,4,2] wanted pixel shift
        use = valid & (E.amin(0) < self.ext_gate_px) & (ext > self.ext_min_ext) & (dd.amin(-1) < 1e5)
        if self.ext_gain > 0 and bool(use.any()):
            x = (cbest[..., 0] - self.img_w / 2) / self.f
            y = (cbest[..., 1] - self.img_h / 2) / self.f
            # unknowns (e_x, e_y, e_z): du/f = e_y - e_z*y ; dv/f = -e_x + e_z*x
            A = torch.zeros(n, K_DET, 8, 3, device=self.dev)
            b = torch.zeros(n, K_DET, 8, device=self.dev)
            A[..., 0::2, 1], A[..., 0::2, 2], b[..., 0::2] = 1.0, -y, res[..., 0] / self.f
            A[..., 1::2, 0], A[..., 1::2, 2], b[..., 1::2] = -1.0, x, res[..., 1] / self.f
            w = use.float()[..., None, None]
            AtA = (A.transpose(-1, -2) @ A * w).sum(1) + 1e-6 * torch.eye(3, device=self.dev)
            Atb = ((A.transpose(-1, -2) @ b[..., None])[..., 0] * w[..., 0]).sum(1)
            de = torch.linalg.solve(AtA, Atb[..., None])[..., 0]
            cnt = use.float().sum(1, keepdim=True)
            self.e = torch.where(cnt > 0, self.e + self.ext_gain * de, self.e).clamp(-self.ext_max, self.ext_max)
        # write the corrected corners back (normalized), same for the corner peaks
        fd[:, :, 8:16] = ((corners / wh) * 2.0 - 1.0).reshape(n, K_DET, 8)
        if peaks is not None and peaks.shape[1] > 0:
            peaks = peaks.clone()
            peaks[..., 2:4] = self._rotate_px(peaks[..., 2:4], self.e)
        return fd.reshape(n, -1), peaks
