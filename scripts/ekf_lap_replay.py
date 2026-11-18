"""Two-lap gate memory in replay: belief error of the next gate at 1-6 m by slot for lap_len 0 (off) and 11 (replica). Run: ~/isaac-env/bin/python -I scripts/ekf_lap_replay.py runs/flight/a25_41/data.pt [...]"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf7 import GateEKF7  # noqa: E402
from droneracing.quad import quat_exp, quat_mul  # noqa: E402

torch.set_num_threads(2)
BASE = dict(sig_map=0.25, kappa=0.3, kappa_corner=0.15, aided_perm=True, n_map=32)
data = [torch.load(f) for f in sys.argv[1:]]


def run(L, d):
    T, N = d["done"].shape
    ekf = GateEKF7(N, "cpu", lap_len=L, **BASE)
    torch.manual_seed(0)

    def reset(ids, t):
        q = quat_mul(d["gtq"][t][ids], quat_exp(torch.randn(len(ids), 3) * 0.02))
        ekf.reset(ids, d["gtp"][t][ids] + torch.randn(len(ids), 3) * 0.05, d["gtv"][t][ids], q)

    reset(torch.arange(N), 0)
    err, ev = {}, []
    for t in range(T):
        ekf.predict(d["imu"][t], 0.01)
        ekf.update(d["feat"][t], d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t])
        vb, _, rel, _ = ekf.outputs(d["gi"][t], d["last"][t], d["mc"][t], d["mn"][t], n_out=1)
        tr = d["gate"][t][:, 0, 0:3] * 5.0
        e = (rel[:, 0] - tr).norm(dim=-1)
        m = d["valid"][t] & (tr.norm(dim=-1) > 1) & (tr.norm(dim=-1) < 6)
        for n in torch.nonzero(m).flatten().tolist():
            err.setdefault(int(d["gi"][t, n]), []).append(float(e[n]))
        ev.append(((vb / 10.0 - d["vel"][t]) * 10.0).norm(dim=-1)[d["valid"][t]])
        if d["done"][t].any() and t + 1 < T:
            reset(torch.nonzero(d["done"][t]).flatten(), t + 1)
    return err, torch.cat(ev)


res = {L: [run(L, d) for d in data] for L in (0, 11)}
print("slot | lap memory off: p50 / p90 | on: p50 / p90")
for g in range(11, 22):
    row = []
    for L in (0, 11):
        v = torch.tensor([x for r, _ in res[L] for x in r.get(g, [])] or [0.0])
        row.append(f"{float(v.median()):.2f} / {float(v.quantile(0.9)):.2f}")
    print(f"{g:3d} | {row[0]} | {row[1]}", flush=True)
for L in (0, 11):
    ev = torch.cat([e for _, e in res[L]])
    print(f"lap_len {L}: velocity error mean {ev.mean():.3f} m/s")
