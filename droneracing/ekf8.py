"""GateEKF8: corner measurements with known identity (MonoRace QuAdGate output) as tight pixel residuals, on top of the delay compensation of GateEKF5.

Input ``peaks`` [n, M, 5] = (gate slot j among the next J map gates, corner type 0-7, u, v [full-image px], score). Types 0-3 are the inner corners
(opening 1.5 m), 4-7 the outer corners (2.7 m); TL, TR, BR, BL. Every corner gives its own 2-D reprojection residual (Huber weighted), the held gate position
``dg`` moves along the residual (kappa_corner). ``feat`` is not used (pass an empty detection). Runs inside the state roll-back of GateEKF5 (``_pre_update``).
"""
from __future__ import annotations

import math

import torch

from .ekf import skew
from .ekf3 import GateEKF3
from .ekf5 import GateEKF5
from .monorace import TYPE_GEOM
from .quad import quat_to_rot


class GateEKF8(GateEKF5):
    def __init__(self, *a, sig_px_mr: float = 1.5, huber_mr: float = 2.0, kappa_mr: float = 0.15, assoc_mr: float = 60.0, max_range_mr: float = 25.0, joint: bool = False, **kw):
        super().__init__(*a, **kw)
        self.joint = joint  # one correlated measurement per gate (shared gate-position error) instead of scalar updates
        self.sig_px_mr, self.huber_mr, self.kappa_mr, self.assoc_mr, self.max_range_mr = sig_px_mr, huber_mr, kappa_mr, assoc_mr, max_range_mr
        self.geom = torch.tensor(TYPE_GEOM, device=self.dev)  # [8,3] right sign, up sign, half size
        self.mr_updates = torch.zeros(self.n, device=self.dev)

    def _corner_world(self, c, nrm, typ):
        """c [n,3] gate centre, nrm [n,3], typ [n] -> corner [n,3]."""
        g = self.geom[typ]
        right = torch.stack((nrm[:, 1], -nrm[:, 0], torch.zeros_like(nrm[:, 0])), -1)
        up = torch.tensor([0.0, 0.0, 1.0], device=self.dev).expand_as(right)
        return c + g[:, 2:3] * (g[:, 0:1] * right + g[:, 1:2] * up)

    @torch.no_grad()
    def _pre_update(self, feat, gate_idx, map_c, map_n, peaks):
        self.mr_updates.zero_()
        if peaks is None or peaks.shape[1] == 0:
            return feat, None
        if self.joint:
            return feat, self._joint_update(gate_idx, map_c, map_n, peaks)
        n, J = self.n, map_c.shape[1]
        ar = torch.arange(n, device=self.dev)
        # corners of one gate share one gate-position error: with k corners in this frame the common (map) part of every corner variance is
        # multiplied by k (sequential scalar updates then carry the information of one correlated measurement, not k independent ones)
        jr_all = peaks[:, :, 0].long() - gate_idx[:, None]
        v_all = (peaks[:, :, 4] > 0.5) & (jr_all >= 0) & (jr_all < J)
        kc = torch.zeros(n, J, device=self.dev)
        kc.scatter_add_(1, jr_all.clamp(0, J - 1), v_all.float())
        for m in range(peaks.shape[1]):
            jr = peaks[:, m, 0].long() - gate_idx  # absolute slot -> offset from the current next gate
            j = jr.clamp(0, J - 1)
            typ = peaks[:, m, 1].long().clamp(0, 7)
            valid = (peaks[:, m, 4] > 0.5) & (jr >= 0) & (jr < J)
            if not valid.any():
                continue
            uv_m = peaks[:, m, 2:4]
            gi = (gate_idx + j).clamp(max=self.dg.shape[1] - 1)
            c_j = map_c[ar, j] + self.dg[ar, gi]
            Pw = self._corner_world(c_j, map_n[ar, j], typ)
            R = self.rot()
            Pb = (R.transpose(1, 2) @ (Pw - self.p)[..., None])[..., 0]
            Xc = (Pb - self.t_cam) @ self.Rb
            Z = Xc[:, 2].clamp_min(0.3)
            uv_p = torch.stack((self.f * Xc[:, 0] / Z + self.img_w / 2, self.f * Xc[:, 1] / Z + self.img_h / 2), -1)
            r = uv_m - uv_p
            ok = valid & (Xc[:, 2] > 0.3) & (Pb.norm(dim=-1) < self.max_range_mr) & (r.norm(dim=-1) < self.assoc_mr) & ((gate_idx + j) <= (self.dg.shape[1] - 1))
            if not ok.any():
                continue
            Jp = torch.zeros(n, 2, 3, device=self.dev)
            Jp[:, 0, 0] = self.f / Z
            Jp[:, 1, 1] = self.f / Z
            Jp[:, 0, 2] = -self.f * Xc[:, 0] / Z ** 2
            Jp[:, 1, 2] = -self.f * Xc[:, 1] / Z ** 2
            JpRb = Jp @ self.Rb.T
            Rt = R.transpose(1, 2)
            H = torch.zeros(n, 2, 15, device=self.dev)
            H[:, :, 0:3] = -(JpRb @ Rt)
            H[:, :, 6:9] = JpRb @ skew(Pb)
            Jg = JpRb @ Rt  # d uv / d gate corner (world)
            kj = kc[ar, j].clamp_min(1.0)
            s_i = (self.sig_px_mr ** 2 + kj * self.sig_map ** 2 * (Jg ** 2).sum(-1).mean(-1)).sqrt()
            rho = r.norm(dim=-1) / (2.0 ** 0.5 * s_i)
            w = torch.where(rho > self.huber_mr, self.huber_mr / rho.clamp_min(1e-6), torch.ones_like(rho))
            Rm = kj[:, None, None] * self.sig_map ** 2 * (Jg @ Jg.transpose(1, 2)) + torch.diag_embed((self.sig_px_mr ** 2 / w ** 2)[:, None].expand(n, 2))
            self._apply(H, r, Rm, ok)
            self.mr_updates += ok.float()
            # held gate position: move it along the residual of the updated state (min-norm step)
            R = self.rot()
            Pw2 = self._corner_world(map_c[ar, j] + self.dg[ar, gi], map_n[ar, j], typ)
            Pb2 = (R.transpose(1, 2) @ (Pw2 - self.p)[..., None])[..., 0]
            Xc2 = (Pb2 - self.t_cam) @ self.Rb
            Z2 = Xc2[:, 2].clamp_min(0.3)
            uv2 = torch.stack((self.f * Xc2[:, 0] / Z2 + self.img_w / 2, self.f * Xc2[:, 1] / Z2 + self.img_h / 2), -1)
            r2 = uv_m - uv2
            S = Jg @ Jg.transpose(1, 2) + 1e-3 * torch.eye(2, device=self.dev)
            dc = (Jg.transpose(1, 2) @ torch.linalg.solve(S, r2[..., None]))[..., 0]
            upd = ok & (r2.norm(dim=-1) < self.assoc_mr)
            self.dg[ar[upd], gi[upd]] = (self.dg[ar, gi] + self.kappa_mr / kj[:, None] * dc)[upd].clamp(-1.5, 1.5)
        return feat, None

    @torch.no_grad()
    def _joint_update(self, gate_idx, map_c, map_n, peaks):
        """Per gate slot: stack the (up to 8) corner residuals [2 x 8] and update once with R = sig_px^2 I + sig_map^2 Jg Jg^T
        (all corners share the error of the gate position, so the corners carry the geometry of the frame, not 8 independent positions)."""
        n, J = self.n, map_c.shape[1]
        ar = torch.arange(n, device=self.dev)
        typs = torch.arange(8, device=self.dev)
        for j in range(min(J, 2)):
            jr = peaks[:, :, 0].long() - gate_idx[:, None]
            sel = (peaks[:, :, 4] > 0.5) & (jr == j)  # [n,M]
            if not sel.any():
                continue
            # measurement per corner type [n,8,2] and its availability
            uvm = torch.zeros(n, 8, 2, device=self.dev)
            have = torch.zeros(n, 8, dtype=torch.bool, device=self.dev)
            ty = peaks[:, :, 1].long().clamp(0, 7)
            for m in range(peaks.shape[1]):
                mm = sel[:, m]
                uvm[ar[mm], ty[mm, m]] = peaks[mm, m, 2:4]
                have[ar[mm], ty[mm, m]] = True
            gi = (gate_idx + j).clamp(max=self.dg.shape[1] - 1)
            c_j = map_c[:, j] + self.dg[ar, gi]
            R = self.rot()
            Rt = R.transpose(1, 2)
            rs, Hs, Js, oks = [], [], [], []
            for t in range(8):
                Pw = self._corner_world(c_j, map_n[:, j], torch.full((n,), t, device=self.dev, dtype=torch.long))
                Pb = (Rt @ (Pw - self.p)[..., None])[..., 0]
                Xc = (Pb - self.t_cam) @ self.Rb
                Z = Xc[:, 2].clamp_min(0.3)
                uv_p = torch.stack((self.f * Xc[:, 0] / Z + self.img_w / 2, self.f * Xc[:, 1] / Z + self.img_h / 2), -1)
                r = uvm[:, t] - uv_p
                ok = have[:, t] & (Xc[:, 2] > 0.3) & (Pb.norm(dim=-1) < self.max_range_mr) & (r.norm(dim=-1) < self.assoc_mr)
                Jp = torch.zeros(n, 2, 3, device=self.dev)
                Jp[:, 0, 0] = self.f / Z
                Jp[:, 1, 1] = self.f / Z
                Jp[:, 0, 2] = -self.f * Xc[:, 0] / Z ** 2
                Jp[:, 1, 2] = -self.f * Xc[:, 1] / Z ** 2
                JpRb = Jp @ self.Rb.T
                Hh = torch.zeros(n, 2, 15, device=self.dev)
                Hh[:, :, 0:3] = -(JpRb @ Rt)
                Hh[:, :, 6:9] = JpRb @ skew(Pb)
                rs.append(r * ok[:, None]); Hs.append(Hh * ok[:, None, None]); Js.append((JpRb @ Rt) * ok[:, None, None]); oks.append(ok)
            r = torch.cat(rs, 1)  # [n,16]
            Hm = torch.cat(Hs, 1)  # [n,16,15]
            Jg = torch.cat(Js, 1)  # [n,16,3]
            ok = torch.stack(oks, 1)  # [n,8]
            k = ok.sum(1)
            use = k >= 2
            if not use.any():
                continue
            # Huber weights per corner from the individual residual (px) against sig_px and the common part
            rn = r.view(n, 8, 2).norm(dim=-1)
            rho = rn / (2.0 ** 0.5 * (self.sig_px_mr ** 2 + self.sig_map ** 2 * (Jg.view(n, 8, 2, 3) ** 2).sum(-1).mean(-1)).sqrt())
            w = torch.where(rho > self.huber_mr, self.huber_mr / rho.clamp_min(1e-6), torch.ones_like(rho))
            dpx = (self.sig_px_mr ** 2 / w ** 2).repeat_interleave(2, dim=1)  # [n,16]
            dpx = torch.where(ok.repeat_interleave(2, dim=1), dpx, torch.full_like(dpx, 1e6))
            Rm = self.sig_map ** 2 * (Jg @ Jg.transpose(1, 2)) + torch.diag_embed(dpx)
            self._apply(Hm, r, Rm, use)
            self.mr_updates += torch.where(use, k.float(), torch.zeros_like(k, dtype=torch.float))
            # held gate position: least-squares step of the stacked residual of the updated state
            R = self.rot()
            Rt = R.transpose(1, 2)
            r2s, J2s = [], []
            for t in range(8):
                Pw = self._corner_world(map_c[:, j] + self.dg[ar, gi], map_n[:, j], torch.full((n,), t, device=self.dev, dtype=torch.long))
                Pb = (Rt @ (Pw - self.p)[..., None])[..., 0]
                Xc = (Pb - self.t_cam) @ self.Rb
                Z = Xc[:, 2].clamp_min(0.3)
                uv2 = torch.stack((self.f * Xc[:, 0] / Z + self.img_w / 2, self.f * Xc[:, 1] / Z + self.img_h / 2), -1)
                r2s.append((uvm[:, t] - uv2) * ok[:, t, None])
            r2 = torch.cat(r2s, 1)
            A = Jg.transpose(1, 2) @ Jg + 1e-2 * torch.eye(3, device=self.dev)
            dc = (torch.linalg.solve(A, (Jg.transpose(1, 2) @ r2[..., None]))[..., 0])
            self.dg[ar[use], gi[use]] = (self.dg[ar, gi] + self.kappa_mr * dc)[use].clamp(-1.5, 1.5)
        return None

    @torch.no_grad()
    def priors8(self, gate_idx, map_c, map_n, J: int = 2, delay: int = 0):
        """Predicted pixels [n,J,8,2] of the 4 inner + 4 outer corners of the next J gates and their visibility [n,J,8], and the camera roll [n]
        (angle from the image up direction to the world up direction), all from the filter state at the capture time (``delay`` steps ago)."""
        n = self.n
        snap = self.hist[-1 - delay] if (delay > 0 and len(self.hist) > delay) else None
        p = snap["p"] if snap is not None else self.p
        q = snap["q"] if snap is not None else self.q
        R = quat_to_rot(q)
        dg = self.dg_for(gate_idx, gate_idx + J, J)
        uv = torch.zeros(n, J, 8, 2, device=self.dev)
        ok = torch.zeros(n, J, 8, dtype=torch.bool, device=self.dev)
        for j in range(J):
            c_j = map_c[:, j] + dg[:, j]
            for t in range(8):
                Pw = self._corner_world(c_j, map_n[:, j], torch.full((n,), t, device=self.dev, dtype=torch.long))
                Pb = (R.transpose(1, 2) @ (Pw - p)[..., None])[..., 0]
                Xc = (Pb - self.t_cam) @ self.Rb
                Z = Xc[:, 2].clamp_min(0.3)
                uv[:, j, t] = torch.stack((self.f * Xc[:, 0] / Z + self.img_w / 2, self.f * Xc[:, 1] / Z + self.img_h / 2), -1)
                ok[:, j, t] = Xc[:, 2] > 0.3
        up_cam = R[:, 2, :] @ self.Rb
        roll = torch.atan2(up_cam[:, 0], -up_cam[:, 1])
        return uv, ok, roll
