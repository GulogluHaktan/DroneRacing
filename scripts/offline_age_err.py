"""Offline (analytic-detector data) estimator error vs time since episode start, full-length streaming."""
import sys
import torch
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from train_estimator import load
from droneracing.estimator import Estimator, split_out
dev = "cuda"
x, lab, done, valid = load([sys.argv[2]], dev)
ck = torch.load(sys.argv[1], map_location=dev)
net = Estimator(ck.get("hidden", 160)).to(dev); net.load_state_dict(ck["net"]); net.eval()
T, N = done.shape
es = torch.zeros_like(done); es[1:] = done[:-1]; es[0] = True
tt = torch.arange(T, device=dev)[:, None].expand(T, N)
age = (tt - torch.where(es, tt, torch.zeros_like(tt)).cummax(0).values).float() * 0.02
nenv = min(N, 1024)
with torch.no_grad():
    o, _ = net(x[:, :nenv].float().transpose(0, 1))  # [n,T,F] streamed from start
o = o.transpose(0, 1)
vel, up, rel, nrm, sl = split_out(o)
ve = ((vel - lab[:T, :nenv, 0:3].float()) * 10).norm(dim=-1)
ag = age[:, :nenv]
alive = ((done[:, :nenv].long().cumsum(0) - done[:, :nenv].long()) == 0)
m = valid[:, :nenv] & alive & (ag > 0.3)
print("T", T, "episodes starting at 0 only (stream not reset)")
for lo, hi in ((0.3, 2), (2, 5), (5, 8), (8, 12), (12, 30)):
    k = m & (ag >= lo) & (ag < hi)
    if k.sum() > 100: print(f"  age {lo}-{hi}s n {int(k.sum())}: vel err {ve[k].mean():.2f} m/s")
