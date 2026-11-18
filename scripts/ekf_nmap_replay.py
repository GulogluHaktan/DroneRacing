"""Replay of recorded replica flights (2 laps = 22 gate slots) through GateEKF3 with n_map 16 (default) and 32: the held gate corrections dg
have n_map slots; slots above n_map-1 share the last entry (clamp), which corrupts the gates of lap 2. Belief error of the next gate at 1-6 m by slot.
Run: ~/isaac-env/bin/python -I scripts/ekf_nmap_replay.py runs/flight/a25_41/data.pt [...]"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf3 import GateEKF3  # noqa: E402
from droneracing.quad import quat_exp, quat_mul  # noqa: E402

torch.set_num_threads(2)
KW = dict(sig_map=0.25, kappa=0.3, kappa_corner=0.15, aided_perm=True)
files = sys.argv[1:]


def run(n_map, d):
    T, N = d["done"].shape
    ekf = GateEKF3(N, "cpu", n_map=n_map, **KW)
    torch.manual_seed(0)

    def reset(ids, t):
        q = quat_mul(d["gtq"][t][ids], quat_exp(torch.randn(len(ids), 3) * 0.02))
        ekf.reset(ids, d["gtp"][t][ids] + torch.randn(len(ids), 3) * 0.05, d["gtv"][t][ids], q)

    reset(torch.arange(N), 0)
    err = {}
    for t in range(T):
        ekf.predict(d["imu"][t], 0.01)
        ekf.update(d["feat"][t], d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t])
        _, _, rel, _ = ekf.outputs(d["gi"][t], d["last"][t], d["mc"][t], d["mn"][t], n_out=3)
        tr = d["gate"][t][:, 0, 0:3] * 5.0
        dist = tr.norm(dim=-1)
        e = (rel[:, 0] - tr).norm(dim=-1)
        m = d["valid"][t] & (dist > 1) & (dist < 6)
        for n in torch.nonzero(m).flatten().tolist():
            err.setdefault(int(d["gi"][t, n]), []).append(float(e[n]))
        if d["done"][t].any() and t + 1 < T:
            reset(torch.nonzero(d["done"][t]).flatten(), t + 1)
    return err


data = [torch.load(f) for f in files]
res = {nm: [run(nm, d) for d in data] for nm in (16, 32)}
print("slot | n_map 16: p50 / p90 | n_map 32: p50 / p90")
for g in range(8, 21):
    row = []
    for nm in (16, 32):
        v = torch.tensor([x for r in res[nm] for x in r.get(g, [])] or [0.0])
        row.append(f"{float(v.median()):.2f} / {float(v.quantile(0.9)):.2f}")
    print(f"{g:3d} | {row[0]} | {row[1]}", flush=True)
