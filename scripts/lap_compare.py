"""Lap 1 vs lap 2 of the 2025 replica (11 gates per lap): speed at the pass, belief error of the next gate at 1-6 m, share of detections accepted, by gate slot.
Run: ~/isaac-env/bin/python -I scripts/lap_compare.py runs/flight/a25_41/data.pt [...]"""
import sys

import torch

sp, er, pn = {}, {}, {}
for f in sys.argv[1:]:
    d = torch.load(f)
    T, N = d["done"].shape
    gi = d["gi"]
    for t in range(1, T):
        inc = (gi[t] > gi[t - 1]) & ~d["done"][t - 1]
        for n in torch.nonzero(inc).flatten().tolist():
            sp.setdefault(int(gi[t - 1, n]), []).append(float(d["gtv"][t - 1, n].norm()))
    for t in range(T):
        tr = d["gate"][t, :, 0, 0:3] * 5.0
        dist = tr.norm(dim=-1)
        e = ((d["est"][t, :, 6:9] * 5.0 - tr)).norm(dim=-1)
        m = d["valid"][t] & (dist > 1) & (dist < 6)
        for n in torch.nonzero(m).flatten().tolist():
            er.setdefault(int(gi[t, n]), []).append(float(e[n]))
print("physical gate | lap 1: speed, err p50/p90 | lap 2: speed, err p50/p90")
for k in range(11):
    a, b = k, k + 11
    if a in er and b in er and a in sp and b in sp:
        ea, eb = torch.tensor(er[a]), torch.tensor(er[b])
        print(f"{k:3d} | {sum(sp[a])/len(sp[a]):5.1f} m/s  {float(ea.median()):.2f}/{float(ea.quantile(.9)):.2f} | {sum(sp[b])/len(sp[b]):5.1f} m/s  {float(eb.median()):.2f}/{float(eb.quantile(.9)):.2f}")
