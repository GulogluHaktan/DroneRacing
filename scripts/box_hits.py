"""Frame hits of recorded camera-line flights (eval_vision --dump): sphere margin vs oriented-box margin at the last state before the episode ends at a gate plane.
Run: ~/isaac-env/bin/python -I scripts/box_hits.py runs/flight/own_41/data.pt [...]   (tilt = angle of the body z axis from world up, includes pitch)"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.quad import quat_to_rot  # noqa: E402

HALF, RS, ARM, PROP, HZ = 0.75, 0.163, 0.113, 0.075, 0.05
hb = torch.tensor([ARM * 0.7071 + PROP] * 2 + [HZ])
rows = []
for f in sys.argv[1:]:
    d = torch.load(f)
    T, N = d["done"].shape
    R = quat_to_rot(d["gtq"].reshape(-1, 4)).reshape(T, N, 3, 3)
    g = d["gate"][:, :, 0]
    c = (R @ (g[..., 0:3] * 5)[..., None])[..., 0]
    n = (R @ g[..., 3:6, None])[..., 0]
    for t, b in torch.nonzero(d["done"][:-1]).tolist():
        s = -(c[t, b] * n[t, b]).sum().item()
        if not (-0.4 < s < 0.0):
            continue
        nn = n[t, b]
        right = torch.stack((nn[1], -nn[0], torch.zeros(()))) / nn[:2].norm()
        lat, ver = (-c[t, b] * right).sum().abs().item(), abs(c[t, b, 2].item())
        Rb = R[t, b]
        el, ev = ((right @ Rb).abs() * hb).sum().item(), (Rb[2].abs() * hb).sum().item()
        rows.append((HALF - RS - max(lat, ver), min(HALF - el - lat, HALF - ev - ver), el, ev, lat, ver))
r = torch.tensor(rows)
hit = r[:, 0] < 0
h = r[hit]
print(f"episode ends at a gate plane {len(r)}; sphere frame hits {len(h)}")
print(f"hits: vertical-dominated {int((h[:,5] >= h[:,4]).sum())}, lateral-dominated {int((h[:,5] < h[:,4]).sum())}")
print(f"hits: sphere margin p50 {h[:,0].median():.3f} m, box margin p50 {h[:,1].median():.3f} m; grazes (sphere margin > -5 cm) {int((h[:,0] > -0.05).sum())}, > -10 cm {int((h[:,0] > -0.10).sum())}")
print(f"box extent at the hits: lateral {h[:,2].mean():.3f} m, vertical {h[:,3].mean():.3f} m (sphere radius {RS} m)")
