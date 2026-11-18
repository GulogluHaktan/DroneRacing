"""State / gate estimator: GateNet detections + IMU + previous command -> what the racing policy needs.

Recurrent (GRU) over the 50 Hz policy steps, trained supervised from simulator ground truth. It plays the role of
the EKF + PnP stage of a MonoRace-style stack, but also keeps a short memory of gates that left the field of view
and tells the policy whether a gate is currently known (``seen``).

Outputs (all in the body frame, matching the policy observation layout of ``RaceEnv``):
  vel_b/10 (3), world-up in body (3), then for the next two gates: relative position/5 (3), normal (3), seen logit (1).
"""
from __future__ import annotations

import torch
import torch.nn as nn

import math

K_DET = 4  # detections per frame fed to the network
DET_DIM = 16  # score, PnP gate centre in body frame /5 (3), gate normal in body frame (3), size, 4 corners (x, y) in [-1, 1]
CAM_OFF = (0.1, 0.0, 0.03)  # camera position in the body frame (VisionRig)
IMU_DIM = 6
ACT_DIM = 4
N_GATES = 2
STRAP_DIM = 6  # strapdown: world-up in body (3), velocity in body /10 (3)
OUT_DIM = 3 + 3 + N_GATES * 7


def pnp_planar(uv: torch.Tensor, f: float, cx: float, cy: float, half: float):
    """Pose of a square (corners tl,tr,br,bl of side 2*half; plane x right, y up) from its 4 pixel corners.

    uv [M,4,2] -> rotation [M,3,3] (plane axes in camera axes), translation [M,3] (plane centre, camera frame), ok [M].
    Homography by DLT (8x8 solve), decomposed with the known intrinsics, rotation re-orthonormalised by SVD.
    """
    M = uv.shape[0]
    dev = uv.device
    x = (uv[..., 0] - cx) / f
    y = (uv[..., 1] - cy) / f
    X = half * torch.tensor([-1.0, 1.0, 1.0, -1.0], device=dev)
    Y = half * torch.tensor([1.0, 1.0, -1.0, -1.0], device=dev)
    one, zero = torch.ones_like(X), torch.zeros_like(X)
    r1 = torch.stack((X.expand(M, 4), Y.expand(M, 4), one.expand(M, 4), zero.expand(M, 4), zero.expand(M, 4), zero.expand(M, 4),
                      -x * X, -x * Y), -1)
    r2 = torch.stack((zero.expand(M, 4), zero.expand(M, 4), zero.expand(M, 4), X.expand(M, 4), Y.expand(M, 4), one.expand(M, 4),
                      -y * X, -y * Y), -1)
    A = torch.stack((r1, r2), 2).reshape(M, 8, 8)
    b = torch.stack((x, y), 2).reshape(M, 8)
    A = A + 1e-6 * torch.eye(8, device=dev)
    h = torch.linalg.solve(A, b[..., None])[..., 0]
    H = torch.cat((h, torch.ones(M, 1, device=dev)), -1).reshape(M, 3, 3)
    h1, h2, h3 = H[:, :, 0], H[:, :, 1], H[:, :, 2]
    lam = 0.5 * (h1.norm(dim=-1) + h2.norm(dim=-1)).clamp_min(1e-9)
    sgn = torch.where(h3[:, 2] < 0, -1.0, 1.0)[:, None]
    c1, c2, t = sgn * h1 / lam[:, None], sgn * h2 / lam[:, None], sgn * h3 / lam[:, None]
    Rraw = torch.stack((c1, c2, torch.linalg.cross(c1, c2)), -1)
    U, _, Vh = torch.linalg.svd(Rraw)
    R = U @ Vh
    ok = torch.isfinite(R).all(-1).all(-1) & torch.isfinite(t).all(-1) & (t[:, 2] > 0.3) & (t[:, 2] < 60.0)
    return torch.nan_to_num(R), torch.nan_to_num(t), ok


def features_from_points(pts: torch.Tensor, score: torch.Tensor, det: torch.Tensor, H: int, W: int,
                         hfov_deg: float = 130.0, cam_tilt: float = 0.75, half: float = 0.75) -> torch.Tensor:
    """pts [B,K,4,2] px, score [B,K], det [B,K] bool (already sorted nearest first) -> [B, K*DET_DIM]."""
    B, K = det.shape
    dev = pts.device
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    R, t, ok = pnp_planar(pts.reshape(B * K, 4, 2), f, W / 2, H / 2, half)
    s, c = math.sin(cam_tilt), math.cos(cam_tilt)
    Rb = torch.tensor([[0.0, s, c], [-1.0, 0.0, 0.0], [0.0, -c, s]], device=dev)  # camera -> body axes (VisionRig.cam_pose)
    pos_b = (t @ Rb.T + torch.tensor(CAM_OFF, device=dev)) / 5.0
    nrm_b = -(torch.linalg.cross(R[:, :, 0], R[:, :, 1]) @ Rb.T)  # gate normal = -(right x up)
    valid = (det.reshape(-1) & ok).float()[:, None]
    size = ((pts.amax(2) - pts.amin(2)).amax(-1).reshape(-1, 1)) / max(H, W)
    ncorn = (pts / torch.tensor([W, H], device=dev) * 2 - 1).reshape(B * K, 8)
    out = torch.cat((score.reshape(-1, 1), pos_b.clamp(-8, 8), nrm_b, size, ncorn), -1) * valid
    return out.reshape(B, K * DET_DIM)


def det_features(dets: list[torch.Tensor], H: int, W: int, device, **kw) -> torch.Tensor:
    """GateNet ``decode`` output (list of [n_i, 9] rows: score + 8 px coords) -> [B, K*DET_DIM], nearest (largest) first."""
    B = len(dets)
    pts = torch.zeros(B, K_DET, 4, 2, device=device)
    pts[..., :, :] = torch.tensor([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]], device=device) * 10 + torch.tensor([W / 2, H / 2], device=device)
    score = torch.zeros(B, K_DET, device=device)
    det = torch.zeros(B, K_DET, dtype=torch.bool, device=device)
    for b, d in enumerate(dets):
        if len(d) == 0:
            continue
        p = d[:, 1:].view(-1, 4, 2)
        size = (p.amax(1) - p.amin(1)).amax(-1)
        order = size.argsort(descending=True)[:K_DET]
        k = len(order)
        pts[b, :k], score[b, :k], det[b, :k] = p[order], d[order, 0], True
    return features_from_points(pts, score, det, H, W, **kw)


def exp_so3(w: torch.Tensor) -> torch.Tensor:
    """Rotation vectors [N,3] -> rotation matrices [N,3,3] (Rodrigues)."""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    K = torch.zeros(w.shape[0], 3, 3, device=w.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=w.device) + s * K + (1 - c) * (K @ K)


class Strapdown:
    """Inertial dead reckoning from the IMU alone (gyro -> attitude, accelerometer -> velocity) in a frame fixed at
    the start of the episode. Gives the network the integration it cannot learn precisely; vision corrects the drift."""

    def __init__(self, n: int, device, dt: float = 0.02):
        self.dt = dt
        self.R = torch.eye(3, device=device).repeat(n, 1, 1)
        self.v = torch.zeros(n, 3, device=device)

    def reset(self, mask: torch.Tensor) -> None:
        self.R[mask] = torch.eye(3, device=self.R.device)
        self.v[mask] = 0.0

    def step(self, imu: torch.Tensor) -> torch.Tensor:
        """imu [N,6] = (gyro/10, accel/20) as logged by ``RaceEnv.imu_obs`` -> [N,6]."""
        gyro, f = imu[:, :3] * 10.0, imu[:, 3:] * 20.0
        self.R = self.R @ exp_so3(gyro * self.dt)
        a_w = (self.R @ f[..., None])[..., 0]
        a_w[:, 2] -= 9.81
        self.v = self.v + a_w * self.dt
        up_b = self.R[:, 2, :]
        v_b = (self.R.transpose(1, 2) @ self.v[..., None])[..., 0] / 10.0
        return torch.cat((up_b, v_b), -1)


def strapdown_sequence(imu: torch.Tensor, ep_start: torch.Tensor) -> torch.Tensor:
    """imu [T,N,6], ep_start [T,N] bool -> strapdown features [T,N,6] (state reset at every episode start)."""
    T, N = ep_start.shape
    sd = Strapdown(N, imu.device)
    out = torch.zeros(T, N, STRAP_DIM, device=imu.device)
    for t in range(T):
        sd.reset(ep_start[t])
        out[t] = sd.step(imu[t])
    return out


class Estimator(nn.Module):
    in_dim = K_DET * DET_DIM + IMU_DIM + ACT_DIM + STRAP_DIM

    def __init__(self, hidden: int = 160):
        super().__init__()
        self.hidden = hidden
        self.inp = nn.Sequential(nn.Linear(self.in_dim, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, OUT_DIM))

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """x [B, T, in_dim] -> out [B, T, OUT_DIM], final hidden [1, B, hidden]."""
        y, h = self.gru(self.inp(x), h)
        return self.head(y), h


def split_out(o: torch.Tensor):
    vel, up = o[..., 0:3], o[..., 3:6]
    gates = o[..., 6:].reshape(*o.shape[:-1], N_GATES, 7)  # rel(3), normal(3), seen logit
    return vel, up, gates[..., 0:3], gates[..., 3:6], gates[..., 6]
