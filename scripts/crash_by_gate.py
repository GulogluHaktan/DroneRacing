"""Where do episodes end? Histogram of the gate index at the end of the episode (crashes) and the per-gate gate-belief error, from camera-line dumps.
Run: ~/isaac-env/bin/python -I scripts/crash_by_gate.py runs/flight/a25_41/data.pt [...]"""
import sys
from collections import Counter

import torch

crash, tot_pass = Counter(), Counter()
err_by_gate = {}
for f in sys.argv[1:]:
    d = torch.load(f)
    T, N = d["done"].shape
    for t in range(T):
        for n in torch.nonzero(d["done"][t]).flatten().tolist():
            g, last = int(d["gi"][t, n]), int(d["last"][t, n])
            if g <= last:
                crash[g] += 1
    gi = d["gi"]
    for t in range(1, T):
        inc = (gi[t] > gi[t - 1]) & ~d["done"][t - 1]
        for n in torch.nonzero(inc).flatten().tolist():
            tot_pass[int(gi[t - 1, n])] += 1
    # belief error of the next gate at 1-6 m, by gate index
    for t in range(T):
        tr = d["gate"][t, :, 0, 0:3] * 5.0
        dist = tr.norm(dim=-1)
        e = ((d["est"][t, :, 6:9] * 5.0 - tr)).norm(dim=-1)
        m = d["valid"][t] & (dist > 1) & (dist < 6)
        for n in torch.nonzero(m).flatten().tolist():
            err_by_gate.setdefault(int(d["gi"][t, n]), []).append(float(e[n]))
print("gate | passes | episodes ended before passing it | crash share | belief error p50 / p90 at 1-6 m")
for g in sorted(set(crash) | set(tot_pass)):
    e = torch.tensor(err_by_gate.get(g, [0.0]))
    n_end, n_pass = crash.get(g, 0), tot_pass.get(g, 0)
    print(f"{g:4d} | {n_pass:6d} | {n_end:4d} | {100 * n_end / max(1, n_end + n_pass):5.1f} % | {float(e.median()):.2f} / {float(e.quantile(0.9)):.2f} m")
