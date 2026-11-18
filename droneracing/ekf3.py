"""EKF2 plus corner-level measurements for gates whose centre is not detected (partly out of view).

``GateEKF2`` consumes whole-gate detections (4 corners each). A gate that is close and partly out of the image gives no centre
detection, so the filter coasts on the map exactly when the position error matters most. ``GateEKF3.update`` first runs the
EKF2 update on the whole-gate detections and then processes single corner peaks (type, score, u, v; from
``gatenet2.decode_corner_peaks``): each peak is associated with the predicted corner of the same type of the nearest of the next J map
gates (pixel distance), and contributes its own 2-D reprojection residual with a Huber weight. Peaks lying on a corner of a
whole-gate detection of this frame are skipped (already used). Base classes are unchanged.
"""
from __future__ import annotations

import torch

from .ekf import GateEKF, skew
from .ekf2 import GateEKF2
from .estimator import DET_DIM, K_DET


class GateEKF3(GateEKF2):
    def __init__(self, *a, sig_px_corner: float = 3.0, assoc_px_corner: float = 30.0, dup_px: float = 6.0, min_score: float = 0.4, kappa_corner: float = 0.3, base_centre: bool = True, max_range: float = 30.0, **kw):
        super().__init__(*a, **kw)
        self.sig_px_c, self.assoc_c, self.dup_px, self.min_score = sig_px_corner, assoc_px_corner, dup_px, min_score
        self.kappa_c, self.base_centre, self.max_range = kappa_corner, base_centre, max_range
        self.corner_updates = torch.zeros(self.n, device=self.dev)

    @torch.no_grad()
    def mask_decoys(self, feat, peaks, gate_idx, map_c, map_n, prev_c, prev_n):
        """Drop detections that belong to the gate just passed (not a candidate in ``update``), returns (feat, peaks).

        Split-S / double gates: the passed gate hangs 2.7 m above the next one, and its frame was associated with the next gate,
        dragging its held offset to the 1.5 m clamp. A whole-gate detection is dropped when its position (filter pose) is closer
        to the passed gate than to the next one; a corner peak when it is closer in pixels to the passed gate's corner of its type.
        Optional (eval_vision --decoy-prev); update() itself is unchanged."""
        n = self.n
        ar = torch.arange(n, device=self.dev)
        has_prev = gate_idx > 0
        cur = map_c[:, 0] + self.dg[ar, gate_idx.clamp(max=self.dg.shape[1] - 1)]
        prv = prev_c + self.dg[ar, (gate_idx - 1).clamp(0, self.dg.shape[1] - 1)]
        R = self.rot()
        fd = feat.reshape(n, K_DET, DET_DIM).clone()
        meas = self.p[:, None] + (R[:, None] @ (fd[:, :, 1:4] * 5.0)[..., None])[..., 0]  # [n,K,3]
        decoy = has_prev[:, None] & ((meas - prv[:, None]).norm(dim=-1) < (meas - cur[:, None]).norm(dim=-1))
        fd[:, :, 0] = torch.where(decoy & (fd[:, :, 0] > 0), torch.zeros_like(fd[:, :, 0]), fd[:, :, 0])
        if peaks is not None and peaks.shape[1] > 0:
            peaks = peaks.clone()
            uv_c, _, Xc = self._project(self._corners_world(map_c, map_n), R)  # [n,J,4,2]
            uv_p, _, Xp = self._project(self._corners_world(prv[:, None], prev_n[:, None]), R)  # [n,1,4,2]
            for m in range(peaks.shape[1]):
                typ, uv_m = peaks[:, m, 0].long(), peaks[:, m, 2:4]
                sel = typ[:, None, None, None]
                dc = (uv_m[:, None] - uv_c.gather(2, sel.expand(n, uv_c.shape[1], 1, 2))[:, :, 0]).norm(dim=-1)
                dc = torch.where(Xc.gather(2, sel.expand(n, Xc.shape[1], 1, 3))[:, :, 0, 2] > 0.3, dc, torch.full_like(dc, 1e6)).amin(-1)
                dp = (uv_m - uv_p.gather(2, sel.expand(n, 1, 1, 2))[:, 0, 0]).norm(dim=-1)
                dp = torch.where(Xp.gather(2, sel.expand(n, 1, 1, 3))[:, 0, 0, 2] > 0.3, dp, torch.full_like(dp, 1e6))
                peaks[:, m, 1] = torch.where(has_prev & (dp < dc), torch.zeros_like(peaks[:, m, 1]), peaks[:, m, 1])
        return fd.reshape(n, -1), peaks

    @torch.no_grad()
    def update(self, feat, gate_idx, map_c, map_n, last_idx, peaks=None):
        if self.base_centre:  # whole-gate detections through the original loosely coupled update (more accurate than EKF2 in replay)
            GateEKF.update(self, feat, gate_idx, map_c, map_n, last_idx)
        else:
            super().update(feat, gate_idx, map_c, map_n, last_idx)
        self.corner_updates.zero_()
        if peaks is None or peaks.shape[1] == 0:
            return
        n, J = self.n, map_c.shape[1]
        ar = torch.arange(n, device=self.dev)
        wh = torch.tensor([self.img_w, self.img_h], device=self.dev)
        fd = feat.reshape(n, K_DET, DET_DIM)
        det_px = (fd[:, :, 8:16].reshape(n, K_DET, 4, 2) + 1.0) * 0.5 * wh  # corners of the whole-gate detections
        det_ok = fd[:, :, 0] > 0
        for m in range(peaks.shape[1]):
            typ, sc, uv_m = peaks[:, m, 0].long(), peaks[:, m, 1], peaks[:, m, 2:4]
            valid = sc > self.min_score
            if not valid.any():
                continue
            d = (det_px - uv_m[:, None, None, :]).norm(dim=-1)  # [n,K,4]
            dup = ((d < self.dup_px) & det_ok[:, :, None]).flatten(1).any(-1)
            R = self.rot()
            uv, Pb, Xc = self._project(self._corners_world(map_c, map_n), R)  # [n,J,4,2]
            sel = typ[:, None, None, None].expand(n, J, 1, 2)
            uv_c = uv.gather(2, sel)[:, :, 0]  # [n,J,2] predicted corner of the measured type
            Pb_c = Pb.gather(2, typ[:, None, None, None].expand(n, J, 1, 3))[:, :, 0]
            Xc_c = Xc.gather(2, typ[:, None, None, None].expand(n, J, 1, 3))[:, :, 0]
            r_all = uv_m[:, None] - uv_c
            err = r_all.norm(dim=-1)
            err = torch.where((Xc_c[..., 2] > 0.3) & (Pb_c.norm(dim=-1) < self.max_range), err, torch.full_like(err, 1e6))  # far corners are tiny and ambiguous: not used
            best = err.argmin(-1)
            ok = valid & ~dup & (err[ar, best] < self.assoc_c) & ((gate_idx + best) <= last_idx)
            if not ok.any():
                continue
            r, Pb_b, Xc_b = r_all[ar, best], Pb_c[ar, best], Xc_c[ar, best]
            Z = Xc_b[:, 2].clamp_min(0.3)
            Jp = torch.zeros(n, 2, 3, device=self.dev)
            Jp[:, 0, 0] = self.f / Z
            Jp[:, 1, 1] = self.f / Z
            Jp[:, 0, 2] = -self.f * Xc_b[:, 0] / Z ** 2
            Jp[:, 1, 2] = -self.f * Xc_b[:, 1] / Z ** 2
            JpRb = Jp @ self.Rb.T
            Rt = R.transpose(1, 2)
            H = torch.zeros(n, 2, 15, device=self.dev)
            H[:, :, 0:3] = -(JpRb @ Rt)
            H[:, :, 6:9] = JpRb @ skew(Pb_b)
            Jg = JpRb @ Rt  # d uv / d gate centre (world)
            s_i = (self.sig_px_c ** 2 + self.sig_map ** 2 * (Jg ** 2).sum(-1).mean(-1)).sqrt()
            rho = r.norm(dim=-1) / (2.0 ** 0.5 * s_i)
            w = torch.where(rho > self.huber, self.huber / rho.clamp_min(1e-6), torch.ones_like(rho))
            Rm = self.sig_map ** 2 * (Jg @ Jg.transpose(1, 2)) + torch.diag_embed((self.sig_px_c ** 2 / w ** 2)[:, None].expand(n, 2))
            self._apply(H, r, Rm, ok)
            self.corner_updates += ok.float()
            # held gate position (dg): move it along the pixel residual of the updated state (minimum-norm step, like the EMA of the base filter)
            R = self.rot()
            gi = (gate_idx + best).clamp(max=self.dg.shape[1] - 1)
            c_ = map_c[ar, best] + self.dg[ar, gi]
            n_ = map_n[ar, best]
            right = torch.stack((n_[:, 1], -n_[:, 0], torch.zeros_like(n_[:, 0])), -1)
            sg = self.sg[typ]  # [n,2] (right, up) signs of the measured corner
            Pw = c_ + self.half * (sg[:, 0:1] * right + sg[:, 1:2] * torch.tensor([0.0, 0.0, 1.0], device=self.dev))
            Pb2 = (R.transpose(1, 2) @ (Pw - self.p)[..., None])[..., 0]
            Xc2 = (Pb2 - self.t_cam) @ self.Rb
            Z2 = Xc2[:, 2].clamp_min(0.3)
            uv2 = torch.stack((self.f * Xc2[:, 0] / Z2 + self.img_w / 2, self.f * Xc2[:, 1] / Z2 + self.img_h / 2), -1)
            r2 = uv_m - uv2
            S = Jg @ Jg.transpose(1, 2) + 1e-3 * torch.eye(2, device=self.dev)
            dc = (Jg.transpose(1, 2) @ torch.linalg.solve(S, r2[..., None]))[..., 0]
            upd = ok & (r2.norm(dim=-1) < self.assoc_c)
            self.dg[ar[upd], gi[upd]] = (self.dg[ar, gi] + self.kappa_c * dc)[upd].clamp(-1.5, 1.5)
