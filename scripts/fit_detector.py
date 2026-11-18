"""Calibrate the analytic detector of the surrogate (droneracing/detector_model.py) on the real detector in the Isaac loop.

Input: `eval_vision.py --percep-log` files (one row per camera frame and drone: next gate eligible (>= 2 corners in the image), size [px at
410 width], corners in the image, view obliqueness [deg], off-axis angle [deg], body rate [rad/s], distance, whole-gate detection found,
mean corner error [px at 410]). Output: detection probability table over (size, corners in image, view obliqueness, body rate) and the
corner error by size, used by `sim_detections(..., calib=...)`.
Run: ~/isaac-env/bin/python -I scripts/fit_detector.py runs/percep/d11/plog_3*.pt --out runs_host/detcal/g15.pt
"""
import argparse
from pathlib import Path

import torch

SIZE = [0, 10, 15, 25, 40, 60, 100, 160, 1e9]
NIN = [2, 3, 4]
VIEW = [0, 30, 50, 65, 90.1]
OMEGA = [0, 3, 6, 1e9]

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--min-n", type=int, default=30)
ap.add_argument("--usable", action="store_true", help="found = whole-gate detection OR >= 2 corner peaks (the filter uses both; the surrogate simulates whole-gate detections only)")
a = ap.parse_args()
rows = [torch.load(f)["rows"] for f in a.logs]
if a.usable:
    rows = [x for x in rows if x.shape[1] >= 10]
    print(f"--usable: {len(rows)} logs with the corner column")
    rows = [torch.cat((x[:, :7], torch.maximum(x[:, 7:8], x[:, 9:10]), x[:, 8:9], x[:, 7:8]), 1) for x in rows]
else:
    rows = [torch.cat((x[:, :9], x[:, 7:8]), 1) for x in rows]
r = torch.cat(rows)  # columns: elig, size, nin, view, off, omega, dist, found (used), box err, found_box
r = r[r[:, 0] > 0.5]  # eligible frames
size, nin, view, om, found, err = r[:, 1], r[:, 2], r[:, 3], r[:, 5], r[:, 7], r[:, 8]
print(f"eligible frames {len(r)}, found {100 * found.mean():.1f} %")
bi = lambda x, edges: (torch.bucketize(x, torch.tensor(edges[1:-1])))  # noqa: E731
bs, bv, bo = bi(size, SIZE), bi(view, VIEW), bi(om, OMEGA)
bn = (nin.clamp(2, 4) - 2).long()
P = torch.zeros(len(SIZE) - 1, 3, len(VIEW) - 1, len(OMEGA) - 1)
N = torch.zeros_like(P)
for i in range(len(r)):
    P[bs[i], bn[i], bv[i], bo[i]] += found[i]
    N[bs[i], bn[i], bv[i], bo[i]] += 1
# sparse cells: back off to the marginal over body rate, then over view
Pm = P.sum(-1, keepdim=True) / N.sum(-1, keepdim=True).clamp_min(1)
Pmm = P.sum((-1, -2), keepdim=True) / N.sum((-1, -2), keepdim=True).clamp_min(1)
table = torch.where(N >= a.min_n, P / N.clamp_min(1), torch.where(N.sum(-1, keepdim=True) >= a.min_n, Pm.expand_as(P), Pmm.expand_as(P)))
sig = torch.zeros(len(SIZE) - 1)
for k in range(len(SIZE) - 1):
    m = (bs == k) & (r[:, 9] > 0.5)  # corner error only from whole-gate detections
    sig[k] = err[m].median() / 1.25 if m.sum() > 10 else (sig[k - 1] if k else 1.0)
for k in range(len(SIZE) - 1):
    row = " ".join(f"{100 * table[k, j].mean():5.1f}" for j in range(3))
    print(f"size {SIZE[k]:>5}-{SIZE[k + 1]:<5} found by corners 2/3/4: {row} %  | corner sigma {sig[k]:.2f} px (n {int(N[k].sum())})")
for v in range(len(VIEW) - 1):
    m = bv == v
    print(f"view {VIEW[v]}-{VIEW[v + 1]} deg: found {100 * found[m].mean():.1f} % (n {int(m.sum())})")
for o in range(len(OMEGA) - 1):
    m = bo == o
    print(f"body rate {OMEGA[o]}-{OMEGA[o + 1]} rad/s: found {100 * found[m].mean():.1f} % (n {int(m.sum())})")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
# relative factors (view obliqueness, body rate) against the mean rate: they keep the error level of the original surrogate (which matched
# Isaac for g17 / g27 / g32) and add only the dependence the policy can exploit
fv = torch.tensor([float(found[bv == v].mean()) if (bv == v).sum() > 20 else float(found.mean()) for v in range(len(VIEW) - 1)]) / found.mean()
fo = torch.tensor([float(found[bo == o].mean()) if (bo == o).sum() > 20 else float(found.mean()) for o in range(len(OMEGA) - 1)]) / found.mean()
print("relative factor by view:", [round(x, 3) for x in fv.tolist()], " by body rate:", [round(x, 3) for x in fo.tolist()])
torch.save({"size": SIZE, "view": VIEW, "omega": OMEGA, "p": table, "sigma": sig, "n": N, "f_view": fv, "f_omega": fo}, a.out)
print("saved", a.out)
