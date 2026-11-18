"""Build the empirical residual bank (Swift-style observation residual model) from closed-loop dumps.

Input: `isaac/eval_vision.py --ekf --dump <dir>` runs (policy flying on the camera + EKF estimate). For every valid step the error of
what the policy saw is stored: next gate and the gate after (body frame, m), velocity (body, m/s), up vector. Traces stay in time
order per drone so the temporal correlation and the heavy tail of the real pipeline are kept; the torch env replays them by
distance to the next gate (EnvCfg.resid_bank). Later the same format is filled from real flights.
Run: ~/isaac-env/bin/python -I scripts/build_resid_bank.py runs/flight/sw_41/data.pt [...] --out runs_host/resid/bank.pt
"""
import argparse
from pathlib import Path

import torch

EDGES = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]  # distance bins of the next gate [m]

ap = argparse.ArgumentParser()
ap.add_argument("dumps", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--clip", type=float, default=4.0, help="cap of a residual vector norm (diverged filter)")
a = ap.parse_args()

res_all, bin_all = [], []
for p in a.dumps:
    d = torch.load(p, map_location="cpu")
    est, gate, vel, up, valid = d["est"].float(), d["gate"].float(), d["vel"].float(), d["up"].float(), d["valid"].bool()
    valid = valid & ~d["done"].bool() & (est[..., 24] > 0.5)  # GRU-estimator dumps mask unseen gates; the EKF always reports them
    g0 = (est[..., 6:9] - gate[..., 0, 0:3]) * 5.0
    g1 = (est[..., 9:12] - gate[..., 1, 0:3]) * 5.0
    dv = (est[..., 0:3] - vel) * 10.0
    du = est[..., 3:6] - up
    res = torch.cat([v * (a.clip / v.norm(dim=-1, keepdim=True).clamp_min(a.clip)) for v in (g0, g1, dv)] + [du], -1)  # [T,N,12]
    dist = (gate[..., 0, 0:3] * 5.0).norm(dim=-1)
    valid = valid & (dist < 30.0)  # past the last gate the dumped "next gate" is an empty slot far away
    b = torch.bucketize(dist, torch.tensor(EDGES))
    T, N = valid.shape
    for i in range(N):  # one drone: contiguous valid runs become trace segments
        v = valid[:, i]
        if v.sum() < 10:
            continue
        idx = torch.nonzero(v).flatten()
        brk = torch.nonzero(idx[1:] != idx[:-1] + 1).flatten()
        bi = b[idx, i].clone()
        bi[brk] = -1  # last sample of each segment
        bi[-1] = -1
        res_all.append(res[idx, i])
        bin_all.append(bi)
    print(f"{p}: {int(valid.sum())} valid steps", flush=True)
res, bins = torch.cat(res_all), torch.cat(bin_all)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({"res": res, "bin": bins, "edges": torch.tensor(EDGES), "sources": a.dumps}, a.out)
gn = res[:, 0:3].norm(dim=-1)
for k in range(len(EDGES) + 1):
    m = bins == k
    if m.any():
        print(f"bin {k}: n {int(m.sum()):7d} | next-gate err median {float(gn[m].median()):.2f} p90 {float(gn[m].quantile(0.9)):.2f} m", flush=True)
print(f"saved {len(res)} samples -> {a.out}")
