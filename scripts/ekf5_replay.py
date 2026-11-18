"""Replay recorded camera-line flights (eval_vision --dump, delay 0) with an ARTIFICIAL image delay: the detections of step t reach the
filter at step t+d. Compares GateEKF3 (delay ignored) with GateEKF5 (delay compensated) and the no-delay reference.
Metric as in ekf_replay.py: in-plane error of the gate positions the policy would see.
Run: ~/isaac-env/bin/python -I scripts/ekf5_replay.py 5 runs/flight/a25_41/data.pt [...]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf3 import GateEKF3  # noqa: E402
from droneracing.ekf5 import GateEKF5  # noqa: E402
from droneracing.quad import quat_exp, quat_mul  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(2)
DT_, _, DC_ = sys.argv[1].partition(":")
D = int(DT_)  # true image delay; optional ":k" = delay assumed by the filter (mismatch test)
DC = int(DC_) if DC_ else D
files = sys.argv[2:]
dev, DT = "cpu", 0.01
KW = dict(sig_map=0.25, kappa=0.3, kappa_corner=0.15, aided_perm=True)


def replay(make, d, delay, comp):
    T, N = d["done"].shape
    ekf = make(N)
    zero = torch.zeros_like(d["feat"][0])

    def reset(ids, t):
        q = quat_mul(d["gtq"][t][ids], quat_exp(torch.randn(len(ids), 3) * 0.02))
        ekf.reset(ids, d["gtp"][t][ids] + torch.randn(len(ids), 3) * 0.05, d["gtv"][t][ids], q)

    reset(torch.arange(N), 0)
    errs, dists = [], []
    for t in range(T):
        ekf.predict(d["imu"][t], DT)
        f = d["feat"][t - delay] if t >= delay else zero
        if comp:
            ekf.update(f, d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t], delay=DC)
        else:
            ekf.update(f, d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t])
        _, _, rel, _ = ekf.outputs(d["gi"][t], d["last"][t], d["mc"][t], d["mn"][t], n_out=3)
        tr = d["gate"][t][..., 0:3] * 5.0
        nrm = d["gate"][t][..., 3:6]
        e = rel[:, :2] - tr
        e = e - (e * nrm).sum(-1, keepdim=True) * nrm
        errs.append(e.norm(dim=-1)); dists.append(tr.norm(dim=-1))
        if d["done"][t].any() and t + 1 < T:
            reset(torch.nonzero(d["done"][t]).flatten(), t + 1)
    return torch.stack(errs), torch.stack(dists)


def summ(label, E, Dd, d):
    T, N = d["done"].shape
    es = torch.zeros_like(d["done"]); es[1:] = d["done"][:-1]; es[0] = True
    tt = torch.arange(T)[:, None].expand(T, N)
    st = torch.where(es, tt, torch.zeros_like(tt)).cummax(0).values
    m = d["valid"] & ((tt - st) * DT > 0.5)
    row = []
    for lo, hi in ((0, 3), (3, 6), (6, 10)):
        k = m & (Dd[:, :, 0] >= lo) & (Dd[:, :, 0] < hi)
        v = E[:, :, 0][k]
        row.append(f"{lo}-{hi}m med {float(v.median()):.2f} p90 {float(v.quantile(0.9)):.2f}")
    print(f"  {label:34s} " + " | ".join(row), flush=True)


for f in files:
    d = torch.load(f)
    print(f, "delay", D, flush=True)
    summ("no delay (reference)", *replay(lambda n: GateEKF3(n, dev, **KW), d, 0, False), d)
    summ(f"delay {D}, ignored (EKF3)", *replay(lambda n: GateEKF3(n, dev, **KW), d, D, False), d)
    summ(f"delay {D}, compensated as {DC} (EKF5)", *replay(lambda n: GateEKF5(n, dev, **KW), d, D, True), d)
