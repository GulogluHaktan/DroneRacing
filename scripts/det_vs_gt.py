"""Detections in a closed-loop dump against the ground-truth gate: for the next gate (slot 0 of `gate`), project its 4 corners with the camera model
and compare with the detected corner sets in `feat` (order-invariant); also the unaided PnP position of the nearest detection against the true gate.
Run: ~/isaac-env/bin/python -I scripts/det_vs_gt.py runs/flight/seg_21/data.pt 410 308 128 0.785"""
import math
import sys

import torch

f_, W, H, hfov, tilt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
d = torch.load(f_)
fx = (W / 2) / math.tan(math.radians(hfov) / 2)
s_, c_ = math.sin(tilt), math.cos(tilt)
RB = torch.tensor([[0.0, s_, c_], [-1.0, 0.0, 0.0], [0.0, -c_, s_]])
T, N = d["done"].shape
feat = d["feat"].reshape(T, N, 4, -1)
rows = []
for t in range(0, T):
    rel = d["gate"][t, :, 0, 0:3] * 5.0
    nrm = d["gate"][t, :, 0, 3:6]
    up = torch.tensor([0.0, 0.0, 1.0])
    for n in range(N):
        if not bool(d["valid"][t, n]):
            continue
        r, nn = rel[n], nrm[n]
        # gate plane axes in the body frame: up = world up in body frame (d["up"]), right = nn x up
        ub = d["up"][t, n]
        right = torch.linalg.cross(nn, ub); right = right / right.norm().clamp_min(1e-6)
        pts = torch.stack((r - 0.75 * right + 0.75 * ub, r + 0.75 * right + 0.75 * ub, r + 0.75 * right - 0.75 * ub, r - 0.75 * right - 0.75 * ub))
        pc = (pts - torch.tensor((0.1, 0.0, 0.03))) @ RB
        if bool((pc[:, 2] < 0.3).any()):
            continue
        uv = torch.stack((fx * pc[:, 0] / pc[:, 2] + W / 2, fx * pc[:, 1] / pc[:, 2] + H / 2), -1)
        if not bool(((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)).all()):
            continue
        ext = float((uv.amax(0) - uv.amin(0)).amax())
        dist = float(r.norm())
        best, bpos = 1e9, 1e9
        nd = 0
        for k in range(4):
            if feat[t, n, k, 0] <= 0:
                continue
            nd += 1
            c = (feat[t, n, k, 8:16].reshape(4, 2) + 1) * 0.5 * torch.tensor([W, H])
            e = min(float((torch.roll(m, i, 0) - uv).norm(dim=-1).mean()) for m in (c, c[[1, 0, 3, 2]]) for i in range(4))
            if e < best:
                best = e
                bpos = float((feat[t, n, k, 1:4] * 5.0 - r).norm())
        rows.append((dist, ext, best, bpos, nd))
r = torch.tensor(rows)
print(f"{f_}: frames with the next gate fully in the image: {len(r)}")
for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 20)):
    m = (r[:, 0] >= lo) & (r[:, 0] < hi)
    if m.sum() == 0:
        continue
    det = r[m, 2] < 1e8
    ok = r[m, 2] < 0.25 * r[m, 1] + 5
    print(f"dist {lo}-{hi} m: n {int(m.sum())}  any detection {100*float(det.float().mean()):.0f}%  correct gate found {100*float(ok.float().mean()):.0f}%  "
          f"corner err (found) median {float(r[m,2][ok].median()) if ok.any() else float('nan'):.1f} px  unaided PnP position err median {float(r[m,3][ok].median()) if ok.any() else float('nan'):.2f} m")
