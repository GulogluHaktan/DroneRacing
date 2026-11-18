"""Closed-loop estimator error statistics from eval_vision --dump data (policy flying on estimator output)."""
import sys
import torch
ds = [torch.load(p) for p in sys.argv[1:]]
d = {k: torch.cat([x[k] for x in ds], 1) for k in ds[0]}
T, N = d["done"].shape
done = d["done"]
es = torch.zeros_like(done); es[1:] = done[:-1]; es[0] = True
tt = torch.arange(T)[:, None].expand(T, N)
st = torch.where(es, tt, torch.zeros_like(tt)).cummax(0).values
age = (tt - st).float() * 0.02  # s since reset
est, vel, up, gate, valid = d["est"].float(), d["vel"].float(), d["up"].float(), d["gate"].float(), d["valid"]
m = valid & (age > 0.3)
ev = ((est[..., 0:3] - vel) * 10)  # body-frame velocity error [m/s]
ve = ev.norm(dim=-1)
spd = vel.norm(dim=-1) * 10
ue = torch.acos((torch.nn.functional.normalize(est[..., 3:6], dim=-1) * up).sum(-1).clamp(-1, 1)) * 57.3
print(f"steps {int(m.sum())}  episodes {int(es.sum())}")
print("vel err m/s: mean %.2f median %.2f p90 %.2f | up err deg mean %.1f p90 %.1f" % (ve[m].mean(), ve[m].median(), ve[m].quantile(.9), ue[m].mean(), ue[m].quantile(.9)))
print("vel err by speed:")
for lo, hi in ((0, 2), (2, 5), (5, 8), (8, 11), (11, 20)):
    k = m & (spd >= lo) & (spd < hi)
    if k.sum() > 100: print(f"  {lo}-{hi} m/s n {int(k.sum())}: mean {ve[k].mean():.2f} rel {float((ve[k]/spd[k].clamp_min(1)).mean()):.2f} bias-mag {ev[k].mean(0).norm():.2f}")
print("vel err by time since reset:")
for lo, hi in ((0.3, 2), (2, 5), (5, 10), (10, 20), (20, 60)):
    k = m & (age >= lo) & (age < hi)
    if k.sum() > 100: print(f"  {lo}-{hi}s n {int(k.sum())}: vel {ve[k].mean():.2f} m/s | up {ue[k].mean():.1f} deg")
# per-axis std and autocorrelation (lag 1, 10, 25 steps) within valid stretches
e0 = ev - ev[m].mean(0)
for lag in (1, 10, 25, 50):
    a, b = e0[:-lag], e0[lag:]
    k = m[:-lag] & m[lag:]
    print(f"vel err autocorr lag {lag}: {float((a[k]*b[k]).sum()/ (a[k]**2).sum()):.2f}")
print("per-axis vel err std", ev[m].std(0).tolist(), "mean", ev[m].mean(0).tolist())
# gates
est_rel = est[..., 6:12].reshape(T, N, 2, 3) * 5
est_nrm = est[..., 12:18].reshape(T, N, 2, 3)
seen_l = est[..., 18:20] > 0
tr, tn, ts = gate[..., 0:3] * 5, gate[..., 3:6], gate[..., 6] > 0.5
dist = tr.norm(dim=-1)
for j in range(2):
    k = m & seen_l[..., j]
    ge = (est_rel[:, :, j] - tr[:, :, j]).norm(dim=-1)
    na = torch.acos((est_nrm[:, :, j] * tn[:, :, j]).sum(-1).clamp(-1, 1)) * 57.3
    print(f"gate{j} reported-seen frac {float(k.sum()/m.sum()):.2f}; true visible-in-env-sense frac {float((m & ts[..., j]).sum()/m.sum()):.2f}")
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 20), (20, 80)):
        kk = k & (dist[:, :, j] >= lo) & (dist[:, :, j] < hi)
        if kk.sum() > 100: print(f"  gate{j} dist {lo}-{hi} m n {int(kk.sum())}: pos err mean {ge[kk].mean():.2f} med {ge[kk].median():.2f} p90 {ge[kk].quantile(.9):.2f} | normal err med {na[kk].median():.0f} deg p90 {na[kk].quantile(.9):.0f}")
sa = (seen_l == ts)[m].float().mean()
print("seen-flag agreement with env visibility %.2f" % sa)
