"""Raw GateNet+PnP detection quality in closed-loop flight (err_* dumps), by age and body tilt."""
import sys
import torch
ds = [torch.load(p) for p in sys.argv[1:]]
d = {k: torch.cat([x[k] for x in ds], 1) for k in ds[0]}
T, N = d["done"].shape
done = d["done"]
es = torch.zeros_like(done); es[1:] = done[:-1]; es[0] = True
tt = torch.arange(T)[:, None].expand(T, N)
age = (tt - torch.where(es, tt, torch.zeros_like(tt)).cummax(0).values).float() * 0.02
feat = d["feat"].float(); T_, N_, F = feat.shape
fd = feat.reshape(T, N, 4, -1)
valid_slot = fd[..., 0] > 0
pos = fd[..., 1:4] * 5.0
tr = d["gate"].float()[..., 0, 0:3] * 5.0  # next gate, body frame
tilt = torch.acos(d["up"].float()[..., 2].clamp(-1, 1)) * 57.3
dist = tr.norm(dim=-1)
err = (pos - tr[:, :, None]).norm(dim=-1)
err = torch.where(valid_slot, err, torch.full_like(err, 1e6))
best = err.min(-1).values
infront = tr[..., 0] > 0.5
m = d["valid"] & infront & (age > 0.3) & (dist < 15)
anyd = valid_slot.any(-1)
print(f"frames with next gate in front and <15 m: {int(m.sum())}; any detection {float(anyd[m].float().mean()):.2f}; good (<1.5 m) {float((best[m] < 1.5).float().mean()):.2f}")
def rep(name, bins, var):
    print(name)
    for lo, hi in bins:
        k = m & (var >= lo) & (var < hi)
        if k.sum() > 200:
            print(f"  {lo}-{hi}: n {int(k.sum())} any-det {float(anyd[k].float().mean()):.2f} good<1.5m {float((best[k]<1.5).float().mean()):.2f} med-err(good-or-not,det only) {float(best[k & anyd].median()):.2f}")
rep("by age s", ((0.3, 2), (2, 5), (5, 10), (10, 20), (20, 60)), age)
rep("by body tilt deg", ((0, 15), (15, 30), (30, 50), (50, 90), (90, 180)), tilt)
rep("by dist m", ((0, 3), (3, 6), (6, 10), (10, 15)), dist)
