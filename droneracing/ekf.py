"""Map-based error-state EKF (MonoRace / AlphaPilot style): IMU strapdown + gate PnP updates against an approximate track map.

State (map frame, z up): position p, velocity v, attitude q (body -> map), accelerometer bias, gyro bias; error state
[dp, dv, dtheta (body, right perturbation), dba, dbg] (15). The track layout (gate centres and normals) is known up to a map error;
detections are associated to the next map gates by Mahalanobis gating against the predicted gate position (outlier rejection
against the state prior, as in MonoRace). The accepted PnP measurement also refines the gate position held for the policy.

Detection input: ``feat`` [B, K*DET_DIM] from ``estimator.det_features`` / ``detector_model.sim_detections``.
"""
from __future__ import annotations

import torch

from .estimator import DET_DIM, K_DET
from .quad import quat_exp, quat_mul, quat_to_rot

G_VEC = (0.0, 0.0, -9.81)


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack((z, -v[..., 2], v[..., 1], v[..., 2], z, -v[..., 0], -v[..., 1], v[..., 0], z), -1).reshape(*v.shape[:-1], 3, 3)


class GateEKF:
    def __init__(self, n: int, device, n_map: int = 16, sig_acc: float = 0.6, sig_gyro: float = 0.03, sig_pnp: tuple = (0.25, 0.10),
                 sig_map: float = 0.25, chi2: float = 14.0, kappa: float = 0.6, sig_nrm: float = 0.35, n_cand: int = 3,
                 aided: bool = True, sig_aided: tuple = (0.15, 0.045), H: int = 192, W: int = 256, hfov: float = 130.0, cam_tilt: float = 0.75,
                 aided_perm: bool = False):
        self.aided, self.sig_aided, self.aided_perm = aided, sig_aided, aided_perm  # aided_perm: try the 4 corner orders (rolled camera)
        self.img_h, self.img_w, self.hfov, self.cam_tilt = H, W, hfov, cam_tilt
        self.n, self.dev = n, device
        self.sig_acc, self.sig_gyro, self.sig_pnp, self.sig_map = sig_acc, sig_gyro, sig_pnp, sig_map
        self.chi2, self.kappa, self.sig_nrm, self.n_cand = chi2, kappa, sig_nrm, n_cand
        self.g = torch.tensor(G_VEC, device=device)
        self.eye15 = torch.eye(15, device=device)
        self.p = torch.zeros(n, 3, device=device)
        self.v = torch.zeros(n, 3, device=device)
        self.q = torch.zeros(n, 4, device=device)
        self.q[:, 0] = 1.0
        self.ba = torch.zeros(n, 3, device=device)
        self.bg = torch.zeros(n, 3, device=device)
        self.P = torch.zeros(n, 15, 15, device=device)
        self.dg = torch.zeros(n, n_map, 3, device=device)  # per-gate correction of the map position from accepted PnP measurements
        self.accepted = torch.zeros(n, device=device)
        self.p0 = (0.05, 0.1, 0.02, 0.2, 0.01)  # initial std: p, v, theta, ba, bg

    def reset(self, ids: torch.Tensor, p: torch.Tensor, v: torch.Tensor, q: torch.Tensor) -> None:
        self.p[ids], self.v[ids], self.q[ids] = p, v, q
        self.ba[ids] = 0.0
        self.bg[ids] = 0.0
        self.dg[ids] = 0.0
        d = torch.cat([torch.full((3,), s ** 2) for s in self.p0]).to(self.dev)
        self.P[ids] = torch.diag(d)

    def rot(self) -> torch.Tensor:
        return quat_to_rot(self.q)

    @torch.no_grad()
    def predict(self, imu: torch.Tensor, dt: float) -> None:
        w = imu[:, :3] * 10.0 - self.bg
        f_raw = imu[:, 3:] * 20.0
        f = f_raw - self.ba
        R = self.rot()
        a_w = (R @ f[..., None])[..., 0] + self.g
        self.p = self.p + self.v * dt + 0.5 * a_w * dt * dt
        self.v = self.v + a_w * dt
        self.q = quat_mul(self.q, quat_exp(w * dt))
        self.q = self.q / self.q.norm(dim=-1, keepdim=True)
        F = self.eye15.expand(self.n, 15, 15).clone()
        F[:, 0:3, 3:6] = torch.eye(3, device=self.dev) * dt
        F[:, 3:6, 6:9] = -(R @ skew(f)) * dt
        F[:, 3:6, 9:12] = -R * dt
        F[:, 6:9, 6:9] = torch.eye(3, device=self.dev) - skew(w) * dt
        F[:, 6:9, 12:15] = -torch.eye(3, device=self.dev) * dt
        sat = (f_raw.abs().amax(-1) > 0.85 * 16 * 9.81).float()  # accelerometer near its range: readings unreliable
        qa = (self.sig_acc * (1.0 + 25.0 * sat) * dt) ** 2
        Q = torch.zeros(self.n, 15, device=self.dev)
        Q[:, 3:6] = qa[:, None]
        Q[:, 6:9] = (self.sig_gyro * dt) ** 2
        Q[:, 9:12] = (2e-3 * dt) ** 2
        Q[:, 12:15] = (2e-4 * dt) ** 2
        self.P = F @ self.P @ F.transpose(1, 2) + torch.diag_embed(Q)

    def _apply(self, H: torch.Tensor, y: torch.Tensor, Rm: torch.Tensor, accept: torch.Tensor) -> None:
        """H [n,m,15], y [n,m], Rm [n,m,m], accept [n] bool."""
        S = H @ self.P @ H.transpose(1, 2) + Rm
        K = self.P @ H.transpose(1, 2) @ torch.linalg.inv(S)
        dx = (K @ y[..., None])[..., 0] * accept[:, None].float()
        IKH = self.eye15 - K @ H * accept[:, None, None].float()
        self.P = IKH @ self.P @ IKH.transpose(1, 2) + K @ Rm @ K.transpose(1, 2) * accept[:, None, None].float()
        self.p = self.p + dx[:, 0:3]
        self.v = self.v + dx[:, 3:6]
        self.q = quat_mul(self.q, quat_exp(dx[:, 6:9]))
        self.q = self.q / self.q.norm(dim=-1, keepdim=True)
        self.ba = self.ba + dx[:, 9:12]
        self.bg = self.bg + dx[:, 12:15]

    @torch.no_grad()
    def update(self, feat: torch.Tensor, gate_idx: torch.Tensor, map_c: torch.Tensor, map_n: torch.Tensor, last_idx: torch.Tensor) -> None:
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
            zn = fd[:, s, 4:7]
            dist = z.norm(dim=-1)
            R = self.rot()
            if self.aided:  # attitude-aided PnP per candidate gate (filter attitude + map gate yaw), as in MonoRace
                z_j = self._aided_positions(fd[:, s, 8:16].reshape(n, 4, 2), R, map_n)  # [n,J,3]
            rel_w = map_c - self.p[:, None, :]  # [n,J,3]; nominal map only: the absolute anchor of the pose
            zhat = (R.transpose(1, 2)[:, None] @ rel_w[..., None])[..., 0]  # [n,J,3]
            if self.aided:
                sig2 = ((self.sig_aided[0] + self.sig_aided[1] * z_j.norm(dim=-1)) ** 2 + self.sig_map ** 2)[..., None, None]  # [n,J,1,1]
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

    def _aided_positions(self, uv_norm: torch.Tensor, R: torch.Tensor, map_n: torch.Tensor) -> torch.Tensor:
        """Gate centre in the body frame for each candidate map gate, from the 4 detected corners and the filter attitude."""
        import math
        n, J = self.n, map_n.shape[1]
        f = (self.img_w / 2) / math.tan(math.radians(self.hfov) / 2)
        s_, c_ = math.sin(self.cam_tilt), math.cos(self.cam_tilt)
        Rb = torch.tensor([[0.0, s_, c_], [-1.0, 0.0, 0.0], [0.0, -c_, s_]], device=self.dev)
        uv = (uv_norm + 1.0) * 0.5 * torch.tensor([self.img_w, self.img_h], device=self.dev)
        right = torch.stack((map_n[..., 1], -map_n[..., 0], torch.zeros_like(map_n[..., 0])), -1)  # [n,J,3]
        zax = torch.tensor([0.0, 0.0, 1.0], device=self.dev).expand(n, J, 3)
        axes_w = torch.stack((right, zax, -map_n), -1)  # [n,J,3,3] columns: plane x, plane y, plane normal
        R_cg = Rb.T @ (R.transpose(1, 2)[:, None] @ axes_w)  # [n,J,3,3]
        uv_j = uv[:, None].expand(n, J, 4, 2).reshape(n * J, 4, 2)
        Rcg = R_cg.reshape(n * J, 3, 3)
        if getattr(self, "aided_perm", False):  # image corner order != gate corner order when the camera is rolled or the gate is seen mirrored (split-S)
            best, best_r = None, None
            for k in range(8):
                uk = torch.roll(uv_j if k < 4 else uv_j[:, [1, 0, 3, 2]], k % 4, dims=1)
                tk = pnp_known_rotation(uk, Rcg, f, self.img_w / 2, self.img_h / 2, 0.75)
                rk = _reproj_err(uk, Rcg, tk, f, self.img_w / 2, self.img_h / 2, 0.75)
                if best is None:
                    best, best_r = tk, rk
                else:
                    better = rk < best_r
                    best, best_r = torch.where(better[:, None], tk, best), torch.where(better, rk, best_r)
            tc = best.reshape(n, J, 3)
        else:
            tc = pnp_known_rotation(uv_j, Rcg, f, self.img_w / 2, self.img_h / 2, 0.75).reshape(n, J, 3)
        return tc @ Rb.T + torch.tensor((0.1, 0.0, 0.03), device=self.dev)

    def dg_for(self, gate_idx: torch.Tensor, last_idx: torch.Tensor, J: int) -> torch.Tensor:
        idx = torch.stack([torch.minimum(gate_idx + j, last_idx) for j in range(J)], 1).clamp(max=self.dg.shape[1] - 1)
        return self.dg[torch.arange(self.n, device=self.dev)[:, None], idx]

    def outputs(self, gate_idx: torch.Tensor, last_idx: torch.Tensor, map_c: torch.Tensor, map_n: torch.Tensor, n_out: int = 2):
        """Policy-side quantities in the body frame: vel_b [n,3], up [n,3], rel [n,n_out,3] (refined map gate - p), nrm [n,n_out,3]."""
        R = self.rot()
        Rt = R.transpose(1, 2)
        vel_b = (Rt @ self.v[..., None])[..., 0]
        up = R[:, 2, :]
        rel_w = map_c[:, :n_out] + self.dg_for(gate_idx, last_idx, map_c.shape[1])[:, :n_out] - self.p[:, None, :]
        rel = (Rt[:, None] @ rel_w[..., None])[..., 0]
        nrm = (Rt[:, None] @ map_n[:, :n_out][..., None])[..., 0]
        return vel_b, up, rel, nrm


def _reproj_err(uv: torch.Tensor, R_cg: torch.Tensor, t: torch.Tensor, f: float, cx: float, cy: float, half: float) -> torch.Tensor:
    """Mean pixel reprojection error of the 4 gate corners for pose (R_cg, t). uv [M,4,2] -> [M]."""
    dev = uv.device
    o = half * torch.tensor([[-1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0]], device=dev)
    P = torch.einsum("mij,kj->mki", R_cg, o) + t[:, None]
    z = P[..., 2].clamp_min(1e-3)
    pr = torch.stack((f * P[..., 0] / z + cx, f * P[..., 1] / z + cy), -1)
    return (pr - uv).norm(dim=-1).mean(-1) + 1e4 * (P[..., 2] < 0.1).any(-1).float()


def pnp_known_rotation(uv: torch.Tensor, R_cg: torch.Tensor, f: float, cx: float, cy: float, half: float) -> torch.Tensor:
    """Gate centre in the camera frame from 4 corner pixels when the gate orientation in the camera frame is known
    (EKF attitude + map yaw), as in MonoRace (PnP with the attitude from the filter). Closed-form least squares over the
    corner rays: t = -(sum M_i)^-1 sum M_i a_i with M_i = I - r_i r_i^T, a_i = R_cg o_i.  uv [M,4,2], R_cg [M,3,3] -> [M,3]."""
    dev = uv.device
    r = torch.stack(((uv[..., 0] - cx) / f, (uv[..., 1] - cy) / f, torch.ones_like(uv[..., 0])), -1)
    r = r / r.norm(dim=-1, keepdim=True)
    Xc = half * torch.tensor([-1.0, 1.0, 1.0, -1.0], device=dev)
    Yc = half * torch.tensor([1.0, 1.0, -1.0, -1.0], device=dev)
    o = torch.stack((Xc, Yc, torch.zeros_like(Xc)), -1)  # [4,3]
    a = torch.einsum("mij,kj->mki", R_cg, o)  # [M,4,3]
    Mi = torch.eye(3, device=dev) - r[..., :, None] * r[..., None, :]  # [M,4,3,3]
    A = Mi.sum(1)
    b = -(Mi @ a[..., None]).sum(1)[..., 0]
    return torch.linalg.solve(A + 1e-6 * torch.eye(3, device=dev), b[..., None])[..., 0]
