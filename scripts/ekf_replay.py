"""Offline replay of the gate EKF variants on recorded closed-loop flights (eval_vision --ekf --dump, new format).

The recorded detections, IMU, map gates and ground truth are fed through each filter variant; the in-plane error of the gate
positions the policy would see is compared (same metric as ekf_gate_err.py). The policy trajectory is NOT re-simulated.
Run: ~/isaac-env/bin/python -I scripts/ekf_replay.py runs/flight/h100r_41/data.pt [...]
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf import GateEKF  # noqa: E402
from droneracing.ekf2 import GateEKF2  # noqa: E402
from droneracing.quad import quat_exp, quat_mul, quat_to_rot  # noqa: E402

torch.manual_seed(0)
files = sys.argv[1:]
dev = "cpu"
DT = 0.01  # h100 policy runs at 100 Hz


def load(p):
    d = torch.load(p)
    return {k: v for k, v in d.items()}


def replay(make, d, label):
    T, N = d["done"].shape
    ekf = make(N)

    def reset(ids, t):
        k = len(ids)
        q = quat_mul(d["gtq"][t][ids], quat_exp(torch.randn(k, 3) * 0.02))
        ekf.reset(ids, d["gtp"][t][ids] + torch.randn(k, 3) * 0.05, d["gtv"][t][ids], q)

    reset(torch.arange(N), 0)
    err_in, dist_all, acc_all = [], [], []
    for t in range(T):
        ekf.predict(d["imu"][t], DT)
        ekf.update(d["feat"][t], d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t])
        _, _, rel, _ = ekf.outputs(d["gi"][t], d["last"][t], d["mc"][t], d["mn"][t], n_out=3)
        est = rel[:, :2]  # [N,2,3] body frame
        tr = d["gate"][t][..., 0:3] * 5.0
        nrm = d["gate"][t][..., 3:6]
        e = est - tr
        e = e - (e * nrm).sum(-1, keepdim=True) * nrm
        err_in.append(e.norm(dim=-1))
        dist_all.append(tr.norm(dim=-1))
        acc_all.append(ekf.accepted.clone())
        done = d["done"][t]
        if done.any() and t + 1 < T:
            reset(torch.nonzero(done).flatten(), t + 1)
    return torch.stack(err_in), torch.stack(dist_all), torch.stack(acc_all), label


def summarize(res, d):
    err, dist, acc, label = res
    T, N = d["done"].shape
    done = d["done"]
    es = torch.zeros_like(done); es[1:] = done[:-1]; es[0] = True
    tt = torch.arange(T)[:, None].expand(T, N)
    st = torch.where(es, tt, torch.zeros_like(tt)).cummax(0).values
    m = d["valid"] & ((tt - st) * DT > 0.5)
    out = {"label": label}
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 30)):
        k = m & (dist[:, :, 0] >= lo) & (dist[:, :, 0] < hi)
        v = err[:, :, 0][k]
        out[f"{lo}-{hi}"] = (float(v.mean()), float(v.median()), float(v.quantile(0.9)), float(acc[k].gt(0).float().mean()))
    k = m & (dist[:, :, 0] < 3)
    out["frac_err>0.5m@<3m"] = float((err[:, :, 0][k] > 0.5).float().mean())
    return out


variants = {
    "V0 baseline (as flown)": lambda n: GateEKF(n, dev, sig_map=0.25),
    "V1 baseline, chi2 14->40": lambda n: GateEKF(n, dev, sig_map=0.25, chi2=40.0),
    "V2 tight corners, Huber": lambda n: GateEKF2(n, dev, sig_map=0.25),
    "V3 tight, sig_px 4": lambda n: GateEKF2(n, dev, sig_map=0.25, sig_px=4.0),
    "V4 tight, min 3 corners": lambda n: GateEKF2(n, dev, sig_map=0.25, min_corners=3),
}
ds = [load(f) for f in files]
only = [a for a in sys.argv[1:] if a.startswith("--only=")]
rows = {}
for name, mk in variants.items():
    agg = None
    errs, dists, accs = [], [], []
    for d in ds:
        e, di, ac, _ = replay(mk, d, name)
        errs.append(e); dists.append(di); accs.append(ac)
    big = {k: torch.cat([x[k] for x in ds], 1) for k in ("done", "valid")}
    res = (torch.cat(errs, 1), torch.cat(dists, 1), torch.cat(accs, 1), name)
    rows[name] = summarize(res, big)
    r = rows[name]
    print(f"{name}", flush=True)
    for b in ("0-3", "3-6", "6-10", "10-15", "15-30"):
        mean, med, p90, af = r[b]
        print(f"   d {b:>5} m: mean {mean:.2f} median {med:.2f} p90 {p90:.2f} | update-accepted {af*100:3.0f}%", flush=True)
    print(f"   steps with in-plane error > 0.5 m while < 3 m from the gate: {r['frac_err>0.5m@<3m']*100:.1f}%", flush=True)
