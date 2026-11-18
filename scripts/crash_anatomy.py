"""Crash anatomy from camera-line dumps: at the closest approach to the next gate before an episode ends, compare the TRUE in-plane miss
of the gate centre with the miss the policy believed (filter estimate), the approach angle and the speed.
Frame-hit crashes = episodes whose last steps pass within 1.5 m of the gate with a true in-plane offset above 0.5 m.
Run: ~/isaac-env/bin/python -I scripts/crash_anatomy.py runs/flight/a25_41/data.pt [...]
"""
import sys

import torch

rows = []
for f in sys.argv[1:]:
    d = torch.load(f)
    T, N = d["done"].shape
    done_t = {}
    for t in range(T):
        for n in torch.nonzero(d["done"][t]).flatten().tolist():
            done_t.setdefault(n, []).append(t)
    for n, ts in done_t.items():
        for t_end in ts:
            lo = max(0, t_end - 25)
            seg = torch.arange(lo, t_end + 1)
            tr = d["gate"][seg, n, 0, 0:3] * 5.0          # true gate centre, body frame
            nr = d["gate"][seg, n, 0, 3:6]                 # true gate normal, body frame
            es = d["est"][seg, n, 6:9] * 5.0               # estimated gate centre, body frame (stored /5 like the gate field)
            dist = tr.norm(dim=-1)
            k = int(dist.argmin())
            if float(dist[k]) > 1.5:
                continue
            off_t = tr[k] - (tr[k] @ nr[k]) * nr[k]
            off_e = es[k] - (es[k] @ nr[k]) * nr[k]
            v = d["vel"][seg[k], n] * 10.0  # body-frame velocity (the gate normal is in the body frame; gtv is in the world frame)
            vb = v.norm()
            ang = torch.rad2deg(torch.acos((v / vb.clamp_min(1e-3) @ (-nr[k] if (tr[k] @ nr[k]) > 0 else nr[k])).abs().clamp(max=1.0))) if vb > 1 else torch.tensor(0.0)
            rows.append((float(off_t.norm()), float(off_e.norm()), float((off_t - off_e).norm()), float(vb), float(ang), float(dist[k])))
r = torch.tensor(rows)
hit = r[:, 0] > 0.5
print(f"episodes ending within 1.5 m of a gate: {len(r)}; with true in-plane miss > 0.5 m (frame hits): {int(hit.sum())}")
if hit.any():
    h = r[hit]
    print("frame hits (mean): true miss %.2f m | believed miss %.2f m | belief error %.2f m | speed %.1f m/s | approach angle %.0f deg" % tuple(h[:, :5].mean(0).tolist()))
    err = h[:, 2]
    print("belief error quantiles (m): p25 %.2f p50 %.2f p75 %.2f p90 %.2f" % tuple(torch.quantile(err, torch.tensor([.25, .5, .75, .9])).tolist()))
    print("share with belief error < 0.2 m (policy aimed fine, still hit): %.0f %%" % (100 * float((err < 0.2).float().mean())))
    print("share with believed miss < 0.45 m (policy thought it was inside): %.0f %%" % (100 * float((h[:, 1] < 0.45).float().mean())))
