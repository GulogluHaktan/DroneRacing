"""Offline sweep of GateEKF5 settings under the OWN camera timing: detections only every 3rd step (30 Hz at 100 Hz policy) and 5 steps late.
Recorded flights (runs/flight/a25_4x, g15 on the replica, GateNet g5, 100 Hz detections) are gated to 30 Hz. Metric: in-plane gate error seen by the policy.
Run: ~/isaac-env/bin/python -I scripts/ekf5_sweep.py runs/flight/a25_41/data.pt [...]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf5 import GateEKF5  # noqa: E402
from droneracing.quad import quat_exp, quat_mul  # noqa: E402

torch.set_num_threads(2)
files = sys.argv[1:]
dev, DT, PER, DELAY = "cpu", 0.01, 3, 5
BASE = dict(sig_map=0.25, kappa=0.3, kappa_corner=0.15, aided_perm=True)
CONFIGS = {
    "base": {},
    "sig_aided (0.10,0.03)": dict(sig_aided=(0.10, 0.03)),
    "sig_aided (0.07,0.02)": dict(sig_aided=(0.07, 0.02)),
    "sig_aided (0.05,0.015)": dict(sig_aided=(0.05, 0.015)),
    "chi2 9 + sig (0.10,0.03)": dict(chi2=9.0, sig_aided=(0.10, 0.03)),
    "chi2 9 + sig (0.07,0.02)": dict(chi2=9.0, sig_aided=(0.07, 0.02)),
    "chi2 6 + sig (0.07,0.02)": dict(chi2=6.0, sig_aided=(0.07, 0.02)),
    "chi2 9 + sig (0.07,0.02) + kappa 0.2": dict(chi2=9.0, sig_aided=(0.07, 0.02), kappa=0.2),
    "chi2 9 + sig (0.07,0.02) + kappa 0.2 + kc 0.1": dict(chi2=9.0, sig_aided=(0.07, 0.02), kappa=0.2, kappa_corner=0.1),
}


def replay(kw, d):
    T, N = d["done"].shape
    ekf = GateEKF5(N, dev, **{**BASE, **kw})
    zero = torch.zeros_like(d["feat"][0])

    def reset(ids, t):
        q = quat_mul(d["gtq"][t][ids], quat_exp(torch.randn(len(ids), 3) * 0.02))
        ekf.reset(ids, d["gtp"][t][ids] + torch.randn(len(ids), 3) * 0.05, d["gtv"][t][ids], q)

    torch.manual_seed(0)
    reset(torch.arange(N), 0)
    errs, dists = [], []
    for t in range(T):
        ekf.predict(d["imu"][t], DT)
        tc = t - DELAY
        f = d["feat"][tc] if (tc >= 0 and tc % PER == 0) else zero
        ekf.update(f, d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t], delay=DELAY)
        _, _, rel, _ = ekf.outputs(d["gi"][t], d["last"][t], d["mc"][t], d["mn"][t], n_out=3)
        tr = d["gate"][t][..., 0:3] * 5.0
        nrm = d["gate"][t][..., 3:6]
        e = rel[:, :2] - tr
        e = e - (e * nrm).sum(-1, keepdim=True) * nrm
        errs.append(e.norm(dim=-1)); dists.append(tr.norm(dim=-1))
        if d["done"][t].any() and t + 1 < T:
            reset(torch.nonzero(d["done"][t]).flatten(), t + 1)
    return torch.stack(errs), torch.stack(dists)


def stats(E, Dd, d):
    T, N = d["done"].shape
    es = torch.zeros_like(d["done"]); es[1:] = d["done"][:-1]; es[0] = True
    tt = torch.arange(T)[:, None].expand(T, N)
    st = torch.where(es, tt, torch.zeros_like(tt)).cummax(0).values
    m = d["valid"] & ((tt - st) * DT > 0.5)
    k = m & (Dd[:, :, 0] < 5)
    v = E[:, :, 0][k]
    return float(v.median()), float(v.quantile(0.9)), float((v > 0.5).float().mean())


data = [torch.load(f) for f in files]
for name, kw in CONFIGS.items():
    rows = [stats(*replay(kw, d), d) for d in data]
    med = sum(r[0] for r in rows) / len(rows); p90 = sum(r[1] for r in rows) / len(rows); fr = sum(r[2] for r in rows) / len(rows)
    print(f"{name:30s} 0-5 m: median {med:.3f}  p90 {p90:.3f}  share>0.5m {fr*100:.1f} %", flush=True)
