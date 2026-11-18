"""Where does the detector miss gates? Held-out own-camera frames (d7 test split of train_gatenet2.py) under the real-camera effects
(droneracing/realcam_aug.py, fixed seed). A gate counts when >= 2 of its labelled corners are in the image; it is found when >= 2 of them
have a corner peak within 4 px (enough for a corner update in the filter). Breakdown by gate size, corners in the image and stress type.
Run: ~/isaac-env/bin/python -I scripts/percep_breakdown.py runs_host/g13/gatenet.pt [--n 1500] [--clean]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.gatenet import GateNet  # noqa: E402
from droneracing.gatenet2 import decode_corner_peaks, make_corner_targets_any  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402
from droneracing.realcam_aug import realcam  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--data", default="runs/percep/d7/data.pt")
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--clean", action="store_true", help="no real-camera effects (appearance augmentation only)")
ap.add_argument("--dev", default="cuda")
ap.add_argument("--thr", type=float, default=0.3)
ap.add_argument("--derotate", type=float, default=-1.0, help=">= 0: derotate with the camera roll + error std [deg] (net trained with --derotate)")
ap.add_argument("--cam-tilt", type=float, default=0.7854)
a = ap.parse_args()
dev = a.dev
data = torch.load(a.data, mmap=True)
H, W = data["hw"]
n = len(data["img"])
perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
te = perm[: max(512, n // 10)][: a.n]
ck = torch.load(a.ckpt, map_location=dev)
net = GateNet(refine=True).to(dev).eval()
net.load_state_dict(ck["net"])
g = torch.Generator(device=dev)
g.manual_seed(0)
rows = []  # (found, size px, corners in image, stress flags)
n_false = n_frames = 0
with torch.no_grad():
    for i in range(0, len(te), 16):
        j = te[i: i + 16]
        img = realistic_augment(data["img"][j].to(dev).float() / 255.0, 1.0, gen=g)
        px = data["corners"][j].to(dev).float() * torch.tensor([W, H], device=dev)
        vis, sv = data["vis"][j].to(dev), data["slot_valid"][j].to(dev)
        dark = torch.zeros(len(j), dtype=torch.bool, device=dev)
        if not a.clean:
            ref = img.mean((1, 2, 3))
            img, px, vis, sv = realcam(img, px, vis, sv, float(data["hfov"]), gen=g, win=data["win"][j] if "win" in data else None, full_hw=data.get("full_hw"))
            dark = img.mean((1, 2, 3)) < 0.6 * ref
        if a.derotate >= 0:
            from droneracing.derotate import derotate_img, derotate_labels, roll_from_quat
            roll = roll_from_quat(data["state"][j][:, 6:10].to(dev).float(), a.cam_tilt) + torch.randn(len(j), generator=g, device=dev) * a.derotate * 3.14159 / 180
            img, px = derotate_img(img, roll), derotate_labels(px, roll, H, W)
        _, _, _, pos = make_corner_targets_any(px, vis, sv, H, W)
        o = net(img)
        peaks = decode_corner_peaks(o[2], o[3], thr=a.thr)
        for b, pk in enumerate(peaks):
            n_frames += 1
            if len(pk):  # peaks farther than 4 px from every labelled corner of the same type
                lab = px[b][pos[b]]  # [L,2]
                typ = torch.nonzero(pos[b])[:, 1]
                for q in pk:
                    mm = typ == int(q[0])
                    if not mm.any() or (lab[mm] - q[2:]).norm(dim=-1).min() >= 4.0:
                        n_false += 1
            for s in range(px.shape[1]):
                npos = int(pos[b, s].sum())
                if npos < 2:
                    continue
                found = found_any = 0
                for c in range(4):
                    if pos[b, s, c]:
                        m = pk[:, 0] == c
                        if m.any() and (pk[m][:, 2:] - px[b, s, c]).norm(dim=-1).min() < 4.0:
                            found += 1
                        if len(pk) and (pk[:, 2:] - px[b, s, c]).norm(dim=-1).min() < 4.0:
                            found_any += 1
                size = float((px[b, s].amax(0) - px[b, s].amin(0)).amax())
                # size in a 410-px-wide full frame (crop px -> full-frame px -> 410 scale), comparable with the Isaac loop numbers
                size *= (float(data["win"][j[b], 2]) / W * 410.0 / data["full_hw"][1]) if "win" in data else 410.0 / W
                rows.append((float(found >= 2), size, npos, float(dark[b]), float(found_any >= 2)))
r = torch.tensor(rows)
print(f"gates {len(r)}  found (>= 2 corners) {100 * r[:, 0].mean():.1f} %  | false corner peaks per frame {n_false / max(1, n_frames):.2f}")
for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 120), (120, 250), (250, 2000)):
    m = (r[:, 1] >= lo) & (r[:, 1] < hi)
    if m.sum():
        print(f"  gate size {lo:4d}-{hi:4d} px: found {100 * r[m, 0].mean():5.1f} %  (share of gates {100 * m.float().mean():4.1f} %, share of misses {100 * (m & (r[:, 0] < 0.5)).sum() / max(1, (r[:, 0] < 0.5).sum()):4.1f} %)")
for k in (2, 3, 4):
    m = r[:, 2] == k
    if m.sum():
        print(f"  corners in image {k}: found {100 * r[m, 0].mean():5.1f} %  (any corner type: {100 * r[m, 4].mean():5.1f} %)  (share of gates {100 * m.float().mean():4.1f} %)")
print(f"  all gates with any corner type accepted: {100 * r[:, 4].mean():.1f} %  (type confusion = the difference)")
for nm, m in (("dark frames", r[:, 3] > 0.5), ("normal frames", r[:, 3] < 0.5)):
    if m.sum():
        print(f"  {nm}: found {100 * r[m, 0].mean():5.1f} %  (n {int(m.sum())})")
