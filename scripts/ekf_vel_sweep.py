"""Velocity and attitude error of the filter against the ground truth for process-noise settings (replay of recorded flights, n_map 32).
Run: ~/isaac-env/bin/python -I scripts/ekf_vel_sweep.py runs/flight/a25_41/data.pt [...]"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf3 import GateEKF3  # noqa: E402
from droneracing.ekf7 import GateEKF7  # noqa: E402
from droneracing.quad import quat_exp, quat_mul  # noqa: E402

torch.set_num_threads(2)
files = sys.argv[1:]
BASE = dict(sig_map=0.25, kappa=0.3, kappa_corner=0.15, aided_perm=True, n_map=32)
CONFIGS = {
    "base": {},
    "ba_walk 0.03": dict(ba_walk=0.03),
    "ba_walk 0.1": dict(ba_walk=0.1),
    "ba_walk 0.3": dict(ba_walk=0.3),
    "ba_walk 1.0": dict(ba_walk=1.0),
    "ba_walk 0.3 + bg_walk 0.01": dict(ba_walk=0.3, bg_walk=0.01),
}


def run(kw, d):
    T, N = d["done"].shape
    ekf = GateEKF7(N, "cpu", **{**BASE, **kw})
    torch.manual_seed(0)

    def reset(ids, t):
        q = quat_mul(d["gtq"][t][ids], quat_exp(torch.randn(len(ids), 3) * 0.02))
        ekf.reset(ids, d["gtp"][t][ids] + torch.randn(len(ids), 3) * 0.05, d["gtv"][t][ids], q)

    reset(torch.arange(N), 0)
    ev, ea = [], []
    for t in range(T):
        ekf.predict(d["imu"][t], 0.01)
        ekf.update(d["feat"][t], d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t])
        vb, up, _, _ = ekf.outputs(d["gi"][t], d["last"][t], d["mc"][t], d["mn"][t], n_out=1)
        ev.append(((vb / 10.0 - d["vel"][t]) * 10.0).norm(dim=-1))
        ea.append(torch.acos((up * d["up"][t]).sum(-1).clamp(-1, 1)) * 57.3)
        if d["done"][t].any() and t + 1 < T:
            reset(torch.nonzero(d["done"][t]).flatten(), t + 1)
    ev, ea = torch.stack(ev), torch.stack(ea)
    m = d["valid"]
    return ev[m], ea[m]


data = [torch.load(f) for f in files]
for name, kw in CONFIGS.items():
    r = [run(kw, d) for d in data]
    ev, ea = torch.cat([x[0] for x in r]), torch.cat([x[1] for x in r])
    print(f"{name:32s} velocity err mean {ev.mean():.3f} p90 {ev.quantile(.9):.3f} m/s | attitude err mean {ea.mean():.2f} p90 {ea.quantile(.9):.2f} deg", flush=True)
