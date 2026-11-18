"""When / how do vision-stack episodes end: age at termination and estimator state in the seconds before it."""
import sys
import torch
ds = [torch.load(p) for p in sys.argv[1:]]
d = {k: torch.cat([x[k] for x in ds], 1) for k in ds[0]}
T, N = d["done"].shape
done = d["done"]
est, vel, up, gate = d["est"].float(), d["vel"].float(), d["up"].float(), d["gate"].float()
ve = ((est[..., 0:3] - vel) * 10).norm(dim=-1)
spd = vel.norm(dim=-1) * 10
rel0e = est[..., 6:9] * 5; rel0t = gate[..., 0, 0:3] * 5
g0e = (rel0e - rel0t).norm(dim=-1)
dist0 = rel0t.norm(dim=-1)
tilt = torch.acos(up[..., 2].clamp(-1, 1)) * 57.3
rows = []
for e in range(N):
    ends = torch.nonzero(done[:, e]).flatten().tolist()
    s = 0
    for t in ends:
        L = t - s + 1
        if L > 60:
            w = slice(max(s, t - 50), t)  # last second
            w0 = slice(s + 15, s + 65)  # first second after the start transient
            rows.append((L * 0.02, float(ve[w, e].mean()), float(ve[w0, e].mean()), float(g0e[w, e].mean()), float(g0e[w0, e].mean()), float(spd[w, e].mean()), float(tilt[w, e].mean()), float(dist0[w, e].mean())))
        s = t + 1
r = torch.tensor(rows)
print("episodes", len(r))
q = lambda x: [round(float(v), 2) for v in x.quantile(torch.tensor([.1, .25, .5, .75, .9]))]
print("episode length s  p10/25/50/75/90:", q(r[:, 0]))
print("vel err LAST second  :", q(r[:, 1]), " | first second:", q(r[:, 2]))
print("gate0 err LAST second:", q(r[:, 3]), " | first second:", q(r[:, 4]))
print("speed last second    :", q(r[:, 5]), " | tilt last second:", q(r[:, 6]), " | dist to next gate:", q(r[:, 7]))
for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 20), (20, 100)):
    k = (r[:, 0] >= lo) & (r[:, 0] < hi)
    print(f"  ends at {lo}-{hi}s: {int(k.sum())} episodes ({float(k.float().mean())*100:.0f}%) | vel err before end {float(r[k,1].mean()) if k.any() else 0:.2f} m/s | speed {float(r[k,5].mean()) if k.any() else 0:.1f} m/s")
