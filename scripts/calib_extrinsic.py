"""Offline calibration of the camera mount rotation from one flight log (eval_vision --dump), without ground truth.

For each drone: replay GateEKF5 on the logged IMU and detections with K candidate mount rotations at the same time (K filter copies),
correct the detections with each candidate (droneracing/extrinsic.py), and score the robust corner residual between the corrected
detections and the gates that the filter predicts at the capture time. Coarse grid, then a fine grid around the best candidate.

Test mode (--synth DEG): the log has no mount error; a random error per drone (each angle uniform in +-DEG) is put into the logged
detections first, and the result is compared with that known error.
Run: ~/isaac-env/bin/python -I scripts/calib_extrinsic.py runs/flight/own_41/data.pt --synth 3 --drones 6 --steps 1500
"""
import argparse
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf5 import GateEKF5  # noqa: E402
from droneracing.estimator import DET_DIM, K_DET  # noqa: E402
from droneracing.extrinsic import correct_feat, distort_feat, rot_xyz  # noqa: E402
from droneracing.quad import quat_exp, quat_mul, quat_to_rot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--synth", type=float, default=0.0)
ap.add_argument("--drones", type=int, default=24)
ap.add_argument("--steps", type=int, default=1500, help="length of the calibration flight [policy steps, 100 Hz]")
ap.add_argument("--delay", type=int, default=5, help="image delay of the log [steps]; the dump holds the delayed detections")
ap.add_argument("--hfov", type=float, default=128.0)
ap.add_argument("--h", type=int, default=308)
ap.add_argument("--w", type=int, default=410)
ap.add_argument("--tilt", type=float, default=0.7854)
ap.add_argument("--clip", type=float, default=25.0, help="residual cap [px] (robust score)")
ap.add_argument("--kappa", type=float, default=0.0, help="gate-position adaptation of the filter during calibration (0 = map fixed, so the filter cannot hide the mount error in the gate positions)")
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
d = torch.load(a.dump, map_location="cpu")
T = min(a.steps, d["done"].shape[0])
H, W = a.h, a.w
KW = dict(sig_map=0.25, kappa=a.kappa, kappa_corner=a.kappa, aided_perm=True, n_map=32, H=H, W=W, hfov=a.hfov, cam_tilt=a.tilt)
PERMS = [list(p) for p in ([0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2], [3, 2, 1, 0], [2, 1, 0, 3], [1, 0, 3, 2], [0, 3, 2, 1])]


def score(b: int, cand: torch.Tensor, E_true: torch.Tensor | None) -> torch.Tensor:
    """Mean clipped corner residual [K] of drone b for the candidate angles cand [K,3] (deg)."""
    K = len(cand)
    E = rot_xyz(cand.to(dev))
    ekf = GateEKF5(K, dev, **KW)
    g = lambda k, t: d[k][t, b].to(dev)[None].expand(K, *d[k].shape[2:])  # noqa: E731

    def reset(t):
        q = quat_mul(g("gtq", t), quat_exp(torch.zeros(K, 3, device=dev)))
        ekf.reset(torch.arange(K, device=dev), g("gtp", t), g("gtv", t), q)

    reset(0)
    tot = torch.zeros(K, device=dev)
    cnt = torch.zeros(K, device=dev)
    wh = torch.tensor([W, H], device=dev, dtype=torch.float32)
    for t in range(T):
        ekf.predict(g("imu", t), 0.01)
        feat = g("feat", t).float()
        if (feat.reshape(K, K_DET, DET_DIM)[:, :, 0] > 0).any():
            if E_true is not None:
                feat = distort_feat(feat, E_true.expand(K, 3, 3), H, W, a.hfov, a.tilt)
            feat = correct_feat(feat, E, H, W, a.hfov, a.tilt)
            if len(ekf.hist) > a.delay:  # residual against the gates predicted at the capture time
                snap = ekf.hist[-1 - a.delay]
                R = quat_to_rot(snap["q"])
                mc, mn = g("mc", t), g("mn", t)
                cw = ekf._corners_world(mc + ekf.dg_for(g("gi", t), g("last", t), mc.shape[1]), mn)
                Pb = (R.transpose(1, 2)[:, None, None] @ (cw - snap["p"][:, None, None, :])[..., None])[..., 0]
                Xc = (Pb - ekf.t_cam) @ ekf.Rb
                Z = Xc[..., 2].clamp_min(1e-3)
                uv = torch.stack((ekf.f * Xc[..., 0] / Z + W / 2, ekf.f * Xc[..., 1] / Z + H / 2), -1)  # [K,J,4,2]
                fd = feat.reshape(K, K_DET, DET_DIM)
                det = (fd[:, :, 0] > 0)
                pts = (fd[:, :, 8:16].reshape(K, K_DET, 4, 2) + 1.0) * 0.5 * wh
                inimg = ((pts >= 0) & (pts < wh)).all(-1).all(-1) & det  # [K,Kd]
                ok_gate = (Xc[..., 2] > 1.0).all(-1)  # [K,J]
                e = torch.full((K, K_DET), 1e9, device=dev)
                for perm in PERMS:
                    dd = (pts[:, :, None, perm] - uv[:, None]).norm(dim=-1).mean(-1)  # [K,Kd,J]
                    dd = torch.where(ok_gate[:, None], dd, torch.full_like(dd, 1e9))
                    e = torch.minimum(e, dd.amin(-1))
                use = inimg & (e < 1e8)
                tot += torch.where(use, e.clamp(max=a.clip), torch.zeros_like(e)).sum(-1)
                cnt += use.float().sum(-1)
        ekf.update(feat, g("gi", t), g("mc", t), g("mn", t), g("last", t), delay=a.delay)
        if bool(d["done"][t, b]) and t + 1 < T:
            reset(t + 1)
    return tot / cnt.clamp_min(1)


res, ests = [], []
nb = min(a.drones, d["done"].shape[1])
gen = torch.Generator().manual_seed(7)
for b in range(nb):
    true = (torch.rand(3, generator=gen) * 2 - 1) * a.synth if a.synth > 0 else None
    E_true = rot_xyz(true.to(dev))[None] if true is not None else None
    coarse = torch.tensor(list(itertools.product(range(-4, 5), repeat=3)), dtype=torch.float32)  # 1 deg grid, +-4 deg
    s1 = score(b, coarse, E_true)
    best = coarse[s1.argmin().item()]
    fine = best + torch.tensor(list(itertools.product([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75], repeat=3)), dtype=torch.float32)
    s2 = score(b, fine, E_true)
    est = fine[s2.argmin().item()]
    ests.append(est)
    s0 = score(b, torch.zeros(1, 3), E_true)[0].item()
    line = f"drone {b:2d} | estimate {est.tolist()} deg | residual {s2.min().item():.2f} px (no correction {s0:.2f} px)"
    if true is None and "extr_err" in d:  # log from eval_vision --tilt-err: compare with the true mount rotation (angle of the residual rotation)
        Et = d["extr_err"][b].float()
        Ee = rot_xyz(est)
        ang = lambda M: torch.rad2deg(torch.acos(((M.diagonal().sum() - 1) / 2).clamp(-1, 1))).item()  # noqa: E731
        line += f" | true error {ang(Et):.2f} deg -> after correction {ang(Ee.T @ Et):.2f} deg"
        res.append(torch.tensor([ang(Ee.T @ Et)]))
    if true is not None:
        err = (est - true).abs()
        line += f" | true {[round(x, 2) for x in true.tolist()]} | abs error {[round(x, 2) for x in err.tolist()]}"
        res.append(err)
    print(line, flush=True)
if res:
    e = torch.stack(res)
    print(f"mean abs error (per axis, or residual angle for a real log) deg: {e.mean(0).tolist()}  max {e.max().item():.2f}")
if a.out:
    torch.save({"deg": torch.stack(ests), "dump": a.dump}, a.out)
    print(f"saved {a.out}")
