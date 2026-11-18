"""True in-plane offset of the drone from the gate centre at every gate pass (gate index increments) in camera-line dumps.
Shows how close to the frame the policy flies when it passes: margin = 0.75 m half opening - body radius (0.163 m) = 0.587 m.
Run: ~/isaac-env/bin/python -I scripts/pass_offsets.py runs/flight/a25_41/data.pt [...]
"""
import sys

import torch

offs, spd = [], []
for f in sys.argv[1:]:
    d = torch.load(f)
    gi, T = d["gi"], d["gi"].shape[0]
    for t in range(1, T):
        inc = (gi[t] > gi[t - 1]) & ~d["done"][t - 1]
        for n in torch.nonzero(inc).flatten().tolist():
            rel = d["gate"][t - 1, n, 0, 0:3] * 5.0
            nr = d["gate"][t - 1, n, 0, 3:6]
            o = rel - (rel @ nr) * nr
            offs.append(float(o.norm())); spd.append(float(d["gtv"][t - 1, n].norm()))
o = torch.tensor(offs); v = torch.tensor(spd)
q = torch.quantile(o, torch.tensor([.25, .5, .75, .9, .99]))
print(f"passes: {len(o)}  true in-plane offset at the pass: mean {o.mean():.2f} m, p25 {q[0]:.2f} p50 {q[1]:.2f} p75 {q[2]:.2f} p90 {q[3]:.2f} p99 {q[4]:.2f}")
print(f"share of passes with offset > 0.40 m: {100*float((o>0.4).float().mean()):.0f} %, > 0.50 m: {100*float((o>0.5).float().mean()):.0f} %  (frame contact above 0.587 m)")
print(f"speed at the pass: mean {v.mean():.1f} m/s;  offset of the fastest quartile {o[v>v.quantile(0.75)].mean():.2f} m vs slowest quartile {o[v<v.quantile(0.25)].mean():.2f} m")
