"""Gate crossings in recorded camera-line flights (eval_vision --dump): sphere (what the sim used) versus the oriented body box.
Uses the true gate (dump field ``gate`` = body-frame centre /5 and normal) and the true attitude at the crossing.
Run: ~/isaac-env/bin/python -I scripts/box_vs_dump.py runs/flight/own_41/data.pt [...]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.quad import quat_to_rot  # noqa: E402

HALF, R_SPH, ARM, PROP, HZ = 0.75, 0.163, 0.113, 0.075, 0.05
rows = []
for f in sys.argv[1:]:
    d = torch.load(f)
    T, N = d["done"].shape
    R = quat_to_rot(d["gtq"].reshape(-1, 4)).reshape(T, N, 3, 3)
    g = d["gate"][:, :, 0]  # next gate
    c_rel = (R @ (g[..., 0:3] * 5.0)[..., None])[..., 0]  # world, gate centre - position
    nrm = (R @ g[..., 3:6, None])[..., 0]
    hb = torch.tensor([ARM * 0.7071 + PROP, ARM * 0.7071 + PROP, HZ])
    for t in range(T - 1):
        gc = d["gtp"][t] + c_rel[t]
        s0 = -(c_rel[t] * nrm[t]).sum(-1)
        s1 = ((d["gtp"][t + 1] - gc) * nrm[t]).sum(-1)
        ok = (s0 < 0) & (s1 >= 0) & d["valid"][t] & ~d["done"][t]
        for b in torch.nonzero(ok).flatten().tolist():
            a = (-s0[b] / (s1[b] - s0[b])).clamp(0, 1)
            pc = d["gtp"][t][b] + a * (d["gtp"][t + 1][b] - d["gtp"][t][b])
            n_ = nrm[t][b]
            right = torch.stack((n_[1], -n_[0], torch.zeros(()))) / n_[:2].norm().clamp_min(1e-6)
            lat, ver = ((pc - gc[b]) * right).abs().sum(), (pc[2] - gc[b][2]).abs()
            Rb = R[t + 1][b]
            ext_lat = ((right @ Rb).abs() * hb).sum()
            ext_ver = (Rb[2].abs() * hb).sum()
            passed = bool(d["gi"][t + 1][b] > d["gi"][t][b])
            yaw_off = torch.rad2deg(torch.atan2(((Rb[:, 0] * right).sum()), (Rb[:, 0] * n_).sum())).abs()
            roll = torch.rad2deg(torch.acos(Rb[2, 2].clamp(-1, 1)))
            rows.append((passed, float(HALF - R_SPH - max(lat, ver)), float(min(HALF - ext_lat - lat, HALF - ext_ver - ver)), float(ext_lat), float(ext_ver), float(yaw_off), float(roll), float(max(lat, ver))))
r = torch.tensor(rows)
P = r[:, 0] > 0.5
print(f"crossings {len(r)}  passed {int(P.sum())}  frame hits {int((~P).sum())}")
print(f"sphere margin<0 among passed (sim allowed by soft rule?): {(r[P,1]<0).sum().item()}")
bx = r[:, 2]
print(f"passed but box margin<0: {int((P & (bx < 0)).sum())} ({100*float((P & (bx<0)).sum())/max(1,int(P.sum())):.1f} %)   box margin p10/p50 of passed: {r[P,2].quantile(0.1):.3f}/{r[P,2].median():.3f}")
if (~P).any():
    print(f"hits: sphere margin p10/p50/p90 {r[~P,1].quantile(0.1):.3f}/{r[~P,1].median():.3f}/{r[~P,1].quantile(0.9):.3f} | box margin {r[~P,2].quantile(0.1):.3f}/{r[~P,2].median():.3f}/{r[~P,2].quantile(0.9):.3f}")
    print(f"hits: extent lat {r[~P,3].mean():.3f} m (sphere {R_SPH}), ext ver {r[~P,4].mean():.3f}; yaw offset {r[~P,5].median():.0f} deg, roll {r[~P,6].median():.0f} deg  | passed: yaw {r[P,5].median():.0f}, roll {r[P,6].median():.0f}, ext lat {r[P,3].mean():.3f}")
    near = (~P) & (r[:, 1] > -0.05)
    print(f"hits with the sphere margin within 5 cm of the pass (near misses) {int(near.sum())}; of them the box would also fail: {int((near & (bx < 0)).sum())}")
