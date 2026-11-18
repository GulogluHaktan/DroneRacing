"""Tightly coupled variant of the map-based EKF (``ekf.GateEKF`` stays untouched).

Instead of one attitude-aided PnP position per detection, every detected gate corner contributes its own pixel reprojection
residual (ADR-VINS style): a partially visible or partly wrong detection still constrains the pose, and each corner is
re-weighted with a Huber loss instead of rejecting the whole detection with one chi-square test.

Camera model, corner order and body/camera conventions are the ones used by ``GateEKF._aided_positions``:
corners (TL, TR, BR, BL; the detector's order) of the 1.5 m opening, body -> camera rotation ``Rb.T``, camera offset ``CAM_OFF``.
"""
from __future__ import annotations

import math

import torch

from .ekf import GateEKF, skew
from .estimator import CAM_OFF, DET_DIM, K_DET

SIGNS = ((-1.0, 1.0), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))  # (right, up) per corner in DETECTOR order: TL, TR, BR, BL (verified on recorded flights: 3 px median)


class GateEKF2(GateEKF):
    def __init__(self, *a, sig_px: float = 2.5, huber: float = 2.0, min_corners: int = 2, assoc_px: float = 45.0, border: float = 0.97,
                 half: float = 0.75, **kw):
        super().__init__(*a, **kw)
        self.sig_px, self.huber, self.min_corners, self.assoc_px, self.border, self.half = sig_px, huber, min_corners, assoc_px, border, half
        self.f = (self.img_w / 2) / math.tan(math.radians(self.hfov) / 2)
        s_, c_ = math.sin(self.cam_tilt), math.cos(self.cam_tilt)
        self.Rb = torch.tensor([[0.0, s_, c_], [-1.0, 0.0, 0.0], [0.0, -c_, s_]], device=self.dev)  # camera -> body
        self.t_cam = torch.tensor(CAM_OFF, device=self.dev)
        self.sg = torch.tensor(SIGNS, device=self.dev)  # [4,2]

    def _corners_world(self, map_c, map_n):
        right = torch.stack((map_n[..., 1], -map_n[..., 0], torch.zeros_like(map_n[..., 0])), -1)  # [n,J,3]
        up = torch.tensor([0.0, 0.0, 1.0], device=self.dev).expand_as(right)
        return map_c[:, :, None, :] + self.half * (self.sg[None, None, :, 0:1] * right[:, :, None, :] + self.sg[None, None, :, 1:2] * up[:, :, None, :])  # [n,J,4,3]

    def _project(self, Pw, R):
        """Pw [n,J,4,3] -> pixels [n,J,4,2], body points [n,J,4,3], camera depth/coords."""
        Pb = (R.transpose(1, 2)[:, None, None] @ (Pw - self.p[:, None, None, :])[..., None])[..., 0]
        Xc = ((Pb - self.t_cam) @ self.Rb)  # row-vector form of Rb^T (Pb - t)
        Z = Xc[..., 2].clamp_min(1e-3)
        uv = torch.stack((self.f * Xc[..., 0] / Z + self.img_w / 2, self.f * Xc[..., 1] / Z + self.img_h / 2), -1)
        return uv, Pb, Xc

    @torch.no_grad()
    def update(self, feat, gate_idx, map_c, map_n, last_idx):
        n, J = self.n, map_c.shape[1]
        fd = feat.reshape(n, K_DET, DET_DIM)
        ar = torch.arange(n, device=self.dev)
        self.accepted.zero_()
        wh = torch.tensor([self.img_w, self.img_h], device=self.dev)
        for s in range(K_DET):
            valid = fd[:, s, 0] > 0
            if not valid.any():
                continue
            R = self.rot()
            meas_n = fd[:, s, 8:16].reshape(n, 4, 2)
            meas = (meas_n + 1.0) * 0.5 * wh  # [n,4,2] px
            Pw = self._corners_world(map_c, map_n)
            uv, Pb, Xc = self._project(Pw, R)
            r = meas[:, None] - uv  # [n,J,4,2]
            in_front = Xc[..., 2] > 0.3
            seen = (meas_n.abs().amax(-1) < self.border)[:, None] & in_front  # corner not clamped to the image border, in front of the camera
            err = r.norm(dim=-1)  # [n,J,4]
            nv = seen.sum(-1)
            score = torch.where(nv >= self.min_corners, (err * seen).sum(-1) / nv.clamp_min(1), torch.full_like(err[..., 0], 1e6))  # [n,J]
            best = score.argmin(-1)
            ok = valid & (score[ar, best] < self.assoc_px) & ((gate_idx + best) <= last_idx)
            if not ok.any():
                continue
            Pb_b, Xc_b, seen_b, r_b = Pb[ar, best], Xc[ar, best], seen[ar, best], r[ar, best]  # [n,4,3] ...
            Z = Xc_b[..., 2].clamp_min(0.3)
            Jp = torch.zeros(n, 4, 2, 3, device=self.dev)
            Jp[..., 0, 0] = self.f / Z
            Jp[..., 1, 1] = self.f / Z
            Jp[..., 0, 2] = -self.f * Xc_b[..., 0] / Z ** 2
            Jp[..., 1, 2] = -self.f * Xc_b[..., 1] / Z ** 2
            JpRb = Jp @ self.Rb.T  # d uv / d Pb  [n,4,2,3]
            Rt = R.transpose(1, 2)
            H = torch.zeros(n, 4, 2, 15, device=self.dev)
            H[..., 0:3] = -(JpRb @ Rt[:, None])
            H[..., 6:9] = JpRb @ skew(Pb_b)
            Jg = (JpRb @ Rt[:, None]).reshape(n, 8, 3)  # d uv / d (gate centre, world): shared by all four corners
            # Huber re-weighting of each corner from its residual relative to its expected spread
            s_i = (self.sig_px ** 2 + self.sig_map ** 2 * (Jg.reshape(n, 4, 2, 3) ** 2).sum(-1).mean(-1)).sqrt()  # [n,4]
            rho = r_b.norm(dim=-1) / (2.0 ** 0.5 * s_i)
            w = torch.where(rho > self.huber, self.huber / rho.clamp_min(1e-6), torch.ones_like(rho))
            w = torch.where(seen_b, w, torch.full_like(w, 1e-4))
            var = (self.sig_px ** 2) / (w ** 2)  # [n,4]
            Rm = self.sig_map ** 2 * (Jg @ Jg.transpose(1, 2))  # [n,8,8] correlated map error
            Rm = Rm + torch.diag_embed(var.repeat_interleave(2, dim=1))
            self._apply(H.reshape(n, 8, 15), r_b.reshape(n, 8), Rm, ok)
            self.accepted += ok.float()
            # refine the held gate position (same heuristic as the base filter) from the attitude-aided PnP of the updated state
            R = self.rot()
            z_j = self._aided_positions(fd[:, s, 8:16].reshape(n, 4, 2), R, map_n)
            z = z_j[ar, best]
            meas_w = self.p + (R @ z[..., None])[..., 0]
            gi = (gate_idx + best).clamp(max=self.dg.shape[1] - 1)
            cur = self.dg[ar, gi]
            new = (1 - self.kappa) * cur + self.kappa * (meas_w - map_c[ar, best])
            self.dg[ar[ok], gi[ok]] = new[ok].clamp(-1.5, 1.5)
