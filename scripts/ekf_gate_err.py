"""Gate-position error of the map-based EKF seen by the policy (eval_vision --ekf --dump data), vs the sim sensor profile.

Reports the in-plane (lateral+vertical) error of the policy's gate position, which decides whether the drone fits through the opening.
"""
import sys

import torch

ds = [torch.load(p) for p in sys.argv[1:]]
d = {k: torch.cat([x[k] for x in ds], 1) for k in ds[0]}
est, gate, valid, done = d["est"].float(), d["gate"].float(), d["valid"], d["done"]
T, N = done.shape
es = torch.zeros_like(done); es[1:] = done[:-1]; es[0] = True
tt = torch.arange(T)[:, None].expand(T, N)
st = torch.where(es, tt, torch.zeros_like(tt)).cummax(0).values
age = (tt - st).float() * 0.02
m = valid & (age > 0.5)
rel_e = est[..., 6:15].reshape(T, N, 3, 3)[:, :, :2] * 5.0  # estimated gate positions (body frame), gates 0,1
rel_t, nrm_t = gate[..., 0:3] * 5.0, gate[..., 3:6]
dist = rel_t.norm(dim=-1)
e = rel_e - rel_t
en = (e * nrm_t).sum(-1, keepdim=True)
inplane = (e - en * nrm_t).norm(dim=-1)  # lateral+vertical miss of the opening centre
print(f"steps {int(m.sum())} | episodes {int(es.sum())}")
for j in range(2):
    print(f"gate {j}: in-plane error of the gate position the policy sees (sim profile: per-axis std ~ (0.06+0.02 d)*(1+0.8 j) m, i.e. in-plane rms ~1.4x that)")
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 30)):
        k = m & (dist[:, :, j] >= lo) & (dist[:, :, j] < hi)
        if k.sum() < 100:
            continue
        v = inplane[:, :, j][k]
        prof = (0.06 + 0.02 * 0.5 * (lo + hi)) * (1 + 0.8 * j) * 1.41
        print(f"  d {lo:2d}-{hi:2d} m n {int(k.sum()):6d}: mean {v.mean():.2f} median {v.median():.2f} p90 {v.quantile(.9):.2f} m | along-normal mean {en[:, :, j][k].abs().mean():.2f} | sim-profile rms ~{prof:.2f}")
