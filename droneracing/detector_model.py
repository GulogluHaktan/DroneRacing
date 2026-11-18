"""Analytic stand-in for 'camera + GateNet' (used to generate millions of estimator training steps without rendering).

Projects the inner corners of the gates around the target gate with the exact camera model of ``isaac/vision_rig.py``
(pinhole, 130 deg, 192x256, 0.1 m forward / 0.03 m up of the body centre, tilt about body y) and corrupts them like the
measured GateNet behaviour: corner noise growing with size, misses for small / partly visible / low-light gates,
rare false boxes. Output matches ``droneracing.estimator.det_features`` (4 detections x [score, 4 corners], sorted by size).
"""
from __future__ import annotations

import math

import torch

from .estimator import K_DET, features_from_points
from .quad import quat_to_rot

SLOTS = (-1, 0, 1, 2)  # gate slots relative to the target gate (as in VisionRig.WINDOW)


def project_corners(env, H: int = 192, W: int = 256, hfov_deg: float = 130.0):
    """uv [N,S,4,2] (tl,tr,br,bl), camera depth [N,S,4], slot validity [N,S]."""
    dev, n = env.device, env.n
    t = env.cfg.cam_tilt
    Rb = torch.tensor([[0.0, math.sin(t), math.cos(t)], [-1.0, 0.0, 0.0], [0.0, -math.cos(t), math.sin(t)]], device=dev)
    Rw = quat_to_rot(env.quad.quat)
    cpos = env.quad.pos + Rw @ torch.tensor([0.1, 0.0, 0.03], device=dev)
    Rc = Rw @ Rb
    f = (W / 2) / math.tan(math.radians(hfov_deg) / 2)
    ar, last = env.arange, env._last_slot()
    half = (0.5 * env.gate_size)[:, None]
    up = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    uvs, zs, oks = [], [], []
    for rel in SLOTS:
        idx = torch.minimum((env.gate_idx + rel).clamp_min(0), last)
        ok = ((env.gate_idx + rel) >= 0) & (idx < env.n_gates)
        c, yaw = env.gate_pos[ar, idx], env.gate_yaw[ar, idx]
        right = torch.stack((torch.sin(yaw), -torch.cos(yaw), torch.zeros_like(yaw)), -1)
        pts = torch.stack((c - half * right + half * up, c + half * right + half * up,
                           c + half * right - half * up, c - half * right - half * up), 1)
        pc = (Rc.transpose(1, 2) @ (pts - cpos[:, None]).transpose(1, 2)).transpose(1, 2)
        z = pc[..., 2]
        zc = z.clamp_min(1e-3)
        uvs.append(torch.stack((f * pc[..., 0] / zc + W / 2, f * pc[..., 1] / zc + H / 2), -1))
        zs.append(z)
        oks.append(ok)
    return torch.stack(uvs, 1), torch.stack(zs, 1), torch.stack(oks, 1)


def _interp(x: torch.Tensor, xs: list[float], ys: list[float]) -> torch.Tensor:
    """Piecewise-linear lookup (clamped at the ends)."""
    X = torch.tensor(xs, device=x.device)
    Y = torch.tensor(ys, device=x.device)
    i = torch.bucketize(x.clamp(X[0], X[-1]), X).clamp(1, len(xs) - 1)
    w = (x.clamp(X[0], X[-1]) - X[i - 1]) / (X[i] - X[i - 1])
    return Y[i - 1] + w * (Y[i] - Y[i - 1])


def _view_angles(env, S: int):
    """Obliqueness of each window gate (deg between the line of sight and the gate normal) [n,S]."""
    ar, last = env.arange, env._last_slot()
    out = []
    for rel in SLOTS[:S]:
        idx = torch.minimum((env.gate_idx + rel).clamp_min(0), last)
        c, yaw = env.gate_pos[ar, idx], env.gate_yaw[ar, idx]
        nrm = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), -1)
        tog = torch.nn.functional.normalize(c - env.quad.pos, dim=-1)
        out.append(torch.rad2deg(torch.acos((tog * nrm).sum(-1).abs().clamp(max=1.0))))
    return torch.stack(out, 1)


def sim_detections(env, gen: torch.Generator | None = None, H: int = 192, W: int = 256, recall: float = 0.88, noise_scale: float = 1.0,
                   hfov_deg: float = 130.0, calib: dict | None = None) -> torch.Tensor:
    """calib (scripts/fit_detector.py): detection probability measured on the real detector in the Isaac loop by size (px at 410 width),
    corners in the image, view obliqueness and body rate, and the corner error by size; replaces the GateNet v3 curves."""
    dev, n = env.device, env.n
    uv, z, ok = project_corners(env, H, W, hfov_deg)
    S = uv.shape[1]
    ctr = uv.mean(2)
    inside = (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H) & (z > 0.3)
    ext = (uv.amax(2) - uv.amin(2)).amax(-1)  # px extent
    centre_in = (ctr[..., 0] >= 0) & (ctr[..., 0] < W) & (ctr[..., 1] >= 0) & (ctr[..., 1] < H) & (z.min(-1).values > 0.3)
    vis = ok & centre_in & (inside.sum(-1) >= 2) & (ext < 4 * max(H, W))
    if calib is not None and calib.get("crop"):  # adaptive-crop line: the detector sees a window around the next (target) gate only
        t = SLOTS.index(0)
        lo, hi = uv[:, t].amin(1), uv[:, t].amax(1)  # [n,2]
        side = ((hi - lo).amax(-1) * 1.6).clamp(min=calib["crop"] * W / 820.0, max=float(max(H, W)))
        c0 = (lo + hi) / 2
        in_win = ((uv - c0[:, None, None]).abs() <= side[:, None, None, None] / 2).all(-1)  # corner inside the window
        vis = vis & ((in_win & inside).sum(-1) >= 2)
    r = lambda *s: torch.rand(*s, generator=gen, device=dev)  # noqa: E731
    # measured on live Isaac frames with GateNet v3 (isaac/eval_vision.py --diag):
    # detection rate by gate size (px extent) and corner error (px per axis, median/1.25) with heavy tails
    nin = inside.sum(-1)
    p = _interp(ext, [6, 10, 14, 20, 40, 60, 120], [0.02, 0.35, 0.75, 0.93, 0.97, 0.90, 0.72]) * torch.where(nin < 4, 0.8, 1.0)
    rec_ = recall[:, None] if torch.is_tensor(recall) else recall  # per-drone recall (domain randomization) or one value
    p = p * (rec_ / 0.88) * (0.85 + 0.15 * r(n, 1))
    if calib is not None:
        s410 = ext * (410.0 / W)
        bs = torch.bucketize(s410, torch.tensor(calib["size"][1:-1], device=dev))
        bv = torch.bucketize(_view_angles(env, S), torch.tensor(calib["view"][1:-1], device=dev))
        bo = torch.bucketize(env.quad.omega.norm(dim=-1)[:, None].expand(n, S), torch.tensor(calib["omega"][1:-1], device=dev))
        if calib.get("mode") == "factors":  # original (Isaac-matched) error level x measured relative view / body-rate dependence
            p = p * calib["f_view"].to(dev)[bv] * calib["f_omega"].to(dev)[bo]
        else:
            tab = calib["p"].to(dev)
            p = tab[bs, (nin.clamp(2, 4) - 2), bv, bo] * (rec_ / 0.95)
    size_term = _interp(ext, [6, 14, 30, 60], [0.0, 0.7, 1.0, 1.0])
    part_term = 0.6 + 0.4 * (nin.float() / 4)
    if not torch.is_tensor(noise_scale) and noise_scale == 0.0:  # ideal detector: every visible gate, exact corners (upper bound for the camera side)
        p = torch.ones_like(p)
    det = vis & (r(n, S) < p)
    if (torch.is_tensor(noise_scale) or noise_scale > 0.0) and (calib is None or calib.get("mode") == "factors"):  # the full table already contains the lost frames
        det = det & (r(n, 1) > 0.05)  # corrupted frames
    sig = _interp(ext, [8, 15, 30, 60, 100], [3.5, 0.9, 1.2, 1.5, 9.0])
    if calib is not None and calib.get("mode") != "factors":  # measured corner error [px at 410] -> this image
        sig = calib["sigma"].to(dev)[torch.bucketize(ext * (410.0 / W), torch.tensor(calib["size"][1:-1], device=dev))] * (W / 410.0)
    ns_ = noise_scale[:, None] if torch.is_tensor(noise_scale) else noise_scale
    sigma = (sig * (1.0 + 0.5 * (nin < 4).float()) * ns_)[..., None, None]
    noise = torch.randn(n, S, 4, 2, generator=gen, device=dev) * sigma
    shared = torch.randn(n, S, 1, 2, generator=gen, device=dev) * sigma * 0.5  # box-level shift (correlated corner error)
    heavy = (r(n, S, 1, 1) < 0.15).float() * 5.0 * (1.0 if torch.is_tensor(noise_scale) else float(noise_scale > 0.0))
    pts = uv + (noise + shared) * (1.0 + heavy)
    score = (0.45 + 0.5 * size_term * part_term + 0.1 * (r(n, S) - 0.5)).clamp(0.3, 1.0)
    # false boxes
    fp = (r(n) < 0.02) & (torch.is_tensor(noise_scale) or noise_scale > 0.0)
    cx, cy = r(n) * W, r(n) * H
    sz = 10 + 30 * r(n)
    fb = torch.stack((torch.stack((cx - sz, cy - sz), -1), torch.stack((cx + sz, cy - sz), -1),
                      torch.stack((cx + sz, cy + sz), -1), torch.stack((cx - sz, cy + sz), -1)), 1)
    pts = torch.cat((pts, fb[:, None]), 1)
    score = torch.cat((score, 0.3 + 0.3 * r(n, 1)), 1)
    det = torch.cat((det, fp[:, None]), 1)
    ext = torch.cat((ext, 2 * sz[:, None]), 1)
    # keep the K largest detections, nearest (largest) first
    key = torch.where(det, ext, torch.full_like(ext, -1.0))
    order = key.argsort(dim=1, descending=True)[:, :K_DET]
    sel = torch.arange(n, device=dev)[:, None]
    d_sel = det[sel, order]
    return features_from_points(pts[sel, order], score[sel, order], d_sel, H, W, hfov_deg=hfov_deg, cam_tilt=env.cfg.cam_tilt)
