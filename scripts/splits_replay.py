"""Offline replay of EKF variants on the A2RL 2025 replica flights, error of the next gate by track section.

At the split-S lower gate the raw attitude-aided PnP is good (median 0.06 m, p90 0.38 m) but the filter output is not
(median ~0.45 m, p90 ~1.7 m) and the filter accepts half as many detections: the filter drifts in the dive and the chi2 gate
then rejects correct detections. Variants test a wider gate and a re-lock (covariance inflation after repeated rejections).
Base GateEKF update only (the flights used EKF3 corner peaks too, which are not in the dump): compare variants relative to V0.
Run: ~/isaac-env/bin/python -I scripts/splits_replay.py runs/flight/a25_4[1-4]/data.pt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf import GateEKF  # noqa: E402
from droneracing.estimator import DET_DIM, K_DET  # noqa: E402
from droneracing.quad import quat_exp, quat_mul  # noqa: E402

torch.manual_seed(0)
DT = 0.01
KW = dict(sig_map=0.25, H=192, W=256, hfov=130.0, cam_tilt=0.75, kappa=0.3)


class Relock(GateEKF):
    """GateEKF + re-lock: after `n_rej` frames in a row with a valid detection but no accepted update, inflate the position
    (and attitude) covariance so the next correct detection passes the chi2 gate."""

    def __init__(self, *a, n_rej=10, sig_p=0.5, sig_th=0.05, **kw):
        super().__init__(*a, **kw)
        self.n_rej, self.sig_p, self.sig_th = n_rej, sig_p, sig_th
        self.rej = torch.zeros(self.n, dtype=torch.long)

    def update(self, feat, gate_idx, map_c, map_n, last_idx):
        super().update(feat, gate_idx, map_c, map_n, last_idx)
        anyv = (feat.reshape(self.n, K_DET, DET_DIM)[:, :, 0] > 0).any(-1)
        self.rej = torch.where(anyv & (self.accepted == 0), self.rej + 1, torch.where(self.accepted > 0, torch.zeros_like(self.rej), self.rej))
        hit = self.rej >= self.n_rej
        if hit.any():
            i = torch.nonzero(hit).flatten()
            self.P[i, 0:3, 0:3] += torch.eye(3) * self.sig_p ** 2
            self.P[i, 6:9, 6:9] += torch.eye(3) * self.sig_th ** 2
            self.rej[i] = 0

    def reset(self, ids, p, v, q):
        super().reset(ids, p, v, q)
        if hasattr(self, "rej"):
            self.rej[ids] = 0


VARIANTS = {
    "V0 as flown (base update)": lambda n: GateEKF(n, "cpu", **KW),
    "V6 aided PnP, 8 corner orders": lambda n: GateEKF(n, "cpu", aided_perm=True, **KW),
}
_UNUSED = {
    "V1 chi2 14 -> 30": lambda n: GateEKF(n, "cpu", chi2=30.0, **KW),
    "V2 re-lock 10 frames, 0.5 m": lambda n: Relock(n, "cpu", **KW),
    "V3 re-lock 5 frames, 0.5 m": lambda n: Relock(n, "cpu", n_rej=5, **KW),
    "V4 re-lock 10 frames, 1.0 m": lambda n: Relock(n, "cpu", sig_p=1.0, **KW),
    "V5 re-lock 10 + chi2 30": lambda n: Relock(n, "cpu", chi2=30.0, **KW),
}


def replay(make, d):
    T, N = d["done"].shape
    ekf = make(N)

    def reset(ids, t):
        q = quat_mul(d["gtq"][t][ids], quat_exp(torch.randn(len(ids), 3) * 0.02))
        ekf.reset(ids, d["gtp"][t][ids] + torch.randn(len(ids), 3) * 0.05, d["gtv"][t][ids], q)

    reset(torch.arange(N), 0)
    err, acc = [], []
    for t in range(T):
        ekf.predict(d["imu"][t].float(), DT)
        ekf.update(d["feat"][t].float(), d["gi"][t], d["mc"][t].float(), d["mn"][t].float(), d["last"][t])
        _, _, rel, _ = ekf.outputs(d["gi"][t], d["last"][t], d["mc"][t].float(), d["mn"][t].float(), n_out=3)
        err.append((rel[:, 0] - d["gate"][t][:, 0, 0:3].float() * 5.0).norm(dim=-1))
        acc.append(ekf.accepted.clone())
        done = d["done"][t]
        if done.any() and t + 1 < T:
            reset(torch.nonzero(done).flatten(), t + 1)
    return torch.stack(err), torch.stack(acc)


ds = [torch.load(p, map_location="cpu") for p in sys.argv[1:]]
for name, mk in VARIANTS.items():
    E, A, M = {}, {}, {}
    for d in ds:
        e, a = replay(mk, d)
        dist = (d["gate"][..., 0, 0:3].float() * 5.0).norm(dim=-1)
        base = d["valid"].bool() & ~d["done"].bool() & (dist > 2.0) & (dist < 8.0)
        g = d["gi"] % 11
        for key, m in (("gates 1-5", base & (g <= 4)), ("gate 6 (split-S low)", base & (g == 5)), ("gates 7-10", base & (g >= 6) & (g <= 9))):
            E.setdefault(key, []).append(e[m]); A.setdefault(key, []).append(a[m])
    print(name, flush=True)
    for key in E:
        e, a = torch.cat(E[key]), torch.cat(A[key])
        print(f"   {key:22s} 2-8 m: median {e.median():.2f} p90 {e.quantile(0.9):.2f} m | accepted/step {a.mean():.2f}", flush=True)
