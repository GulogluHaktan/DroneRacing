"""When is the velocity estimate wrong? Camera-line dumps: velocity error (body frame) against the accelerometer load, the time since the last accepted gate measurement is not stored,
so the speed and the gate distance are used. Run: ~/isaac-env/bin/python -I scripts/vel_err_anatomy.py runs/flight/a25_41/data.pt [...]"""
import sys

import torch

rows = []
for f in sys.argv[1:]:
    d = torch.load(f)
    v_est = d["est"][..., 0:3] * 10.0
    v_gt = d["vel"] * 10.0
    err = (v_est - v_gt).norm(dim=-1)  # [T,N]
    acc = (d["imu"][..., 3:6] * 20.0).norm(dim=-1)  # specific force magnitude [m/s^2]
    dist = (d["gate"][:, :, 0, 0:3] * 5.0).norm(dim=-1)
    spd = (d["gtv"]).norm(dim=-1)
    m = d["valid"]
    rows.append(torch.stack((err[m], acc[m], dist[m], spd[m]), -1))
r = torch.cat(rows)
print(f"samples {len(r)}  velocity error: mean {r[:,0].mean():.2f}  median {r[:,0].median():.2f}  p90 {r[:,0].quantile(.9):.2f}  p99 {r[:,0].quantile(.99):.2f} m/s")
print("by specific force |f| [m/s^2]:")
for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 100), (100, 400)):
    k = (r[:, 1] >= lo) & (r[:, 1] < hi)
    if k.sum() > 50:
        print(f"  {lo:3d}-{hi:3d}: share {100*float(k.float().mean()):5.1f} %  err mean {r[k,0].mean():.2f}  p90 {r[k,0].quantile(.9):.2f} m/s")
print("by distance to the next gate [m]:")
for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 80)):
    k = (r[:, 2] >= lo) & (r[:, 2] < hi)
    if k.sum() > 50:
        print(f"  {lo:3d}-{hi:3d}: share {100*float(k.float().mean()):5.1f} %  err mean {r[k,0].mean():.2f}  p90 {r[k,0].quantile(.9):.2f} m/s")
print("by speed [m/s]:")
for lo, hi in ((0, 8), (8, 14), (14, 18), (18, 22), (22, 60)):
    k = (r[:, 3] >= lo) & (r[:, 3] < hi)
    if k.sum() > 50:
        print(f"  {lo:3d}-{hi:3d}: share {100*float(k.float().mean()):5.1f} %  err mean {r[k,0].mean():.2f}  p90 {r[k,0].quantile(.9):.2f} m/s")
