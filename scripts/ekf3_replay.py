"""Replay recorded flights through GateEKF (as flown), GateEKF2 and GateEKF3 with SYNTHETIC corner peaks (upper bound of what a
partial-gate corner detector can give): every corner of the next two gates that is inside the image is detected with a
probability that depends on how many corners of the gate are in the image, plus Gaussian pixel noise.
Run: ~/isaac-env/bin/python -I scripts/ekf3_replay.py runs/flight/h100r_41/data.pt [...] [--recall=1.0] [--noise=1.5]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf import GateEKF  # noqa: E402
from droneracing.ekf2 import GateEKF2  # noqa: E402
from droneracing.ekf3 import GateEKF3  # noqa: E402
from droneracing.quad import quat_exp, quat_mul, quat_to_rot  # noqa: E402
from droneracing.ekf2 import SIGNS  # noqa: E402

torch.manual_seed(0)
files = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = {a.split("=")[0][2:]: float(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--")}
RECALL, NOISE = opt.get("recall", 1.0), opt.get("noise", 1.5)
DT, dev = 0.01, "cpu"
SG = torch.tensor(SIGNS)


def synth_peaks(ekf, d, t, W=256, H=192):
    """Corner peaks [N,M,4] (type, score, u, v) from the ground-truth gate corners (next two gates) of step t."""
    R = quat_to_rot(d["gtq"][t])
    pos = d["gtp"][t]
    N = pos.shape[0]
    rows = []
    for k in range(2):
        rel_b = d["gate"][t][:, k, 0:3] * 5.0
        nrm_b = d["gate"][t][:, k, 3:6]
        c_w = pos + (R @ rel_b[..., None])[..., 0]
        n_w = (R @ nrm_b[..., None])[..., 0]
        right = torch.stack((n_w[:, 1], -n_w[:, 0], torch.zeros(N)), -1)
        up = torch.tensor([0.0, 0.0, 1.0]).expand(N, 3)
        Pw = c_w[:, None] + 0.75 * (SG[None, :, 0:1] * right[:, None] + SG[None, :, 1:2] * up[:, None])  # [N,4,3]
        Pb = (R.transpose(1, 2)[:, None] @ (Pw - pos[:, None])[..., None])[..., 0]
        Xc = (Pb - ekf.t_cam) @ ekf.Rb
        Z = Xc[..., 2].clamp_min(1e-3)
        uv = torch.stack((ekf.f * Xc[..., 0] / Z + W / 2, ekf.f * Xc[..., 1] / Z + H / 2), -1)
        inside = (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H) & (Xc[..., 2] > 0.3)
        nin = inside.sum(-1, keepdim=True)
        p = RECALL * torch.where(nin >= 4, 0.95, torch.where(nin == 3, 0.85, torch.where(nin == 2, 0.6, 0.3)))
        det = inside & (torch.rand(N, 4) < p)
        for c in range(4):
            sc = det[:, c].float() * 0.8
            rows.append(torch.stack((torch.full((N,), float(c)), sc, uv[:, c, 0] + torch.randn(N) * NOISE, uv[:, c, 1] + torch.randn(N) * NOISE), -1))
    return torch.stack(rows, 1)


def run(make, d, use_peaks):
    T, N = d["done"].shape
    ekf = make(N)

    def reset(ids, t):
        q = quat_mul(d["gtq"][t][ids], quat_exp(torch.randn(len(ids), 3) * 0.02))
        ekf.reset(ids, d["gtp"][t][ids] + torch.randn(len(ids), 3) * 0.05, d["gtv"][t][ids], q)

    reset(torch.arange(N), 0)
    errs, dists = [], []
    for t in range(T):
        ekf.predict(d["imu"][t], DT)
        if use_peaks:
            ekf.update(d["feat"][t], d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t], peaks=synth_peaks(ekf, d, t))
        else:
            ekf.update(d["feat"][t], d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t])
        _, _, rel, _ = ekf.outputs(d["gi"][t], d["last"][t], d["mc"][t], d["mn"][t], n_out=3)
        tr = d["gate"][t][..., 0:3] * 5.0
        nrm = d["gate"][t][..., 3:6]
        e = rel[:, :2] - tr
        e = e - (e * nrm).sum(-1, keepdim=True) * nrm
        errs.append(e.norm(dim=-1)); dists.append(tr.norm(dim=-1))
        if d["done"][t].any() and t + 1 < T:
            reset(torch.nonzero(d["done"][t]).flatten(), t + 1)
    return torch.stack(errs), torch.stack(dists)


ds = [torch.load(f) for f in files]
variants = [("EKF as flown", lambda n: GateEKF(n, dev, sig_map=0.25), False),
            ("EKF2 tight", lambda n: GateEKF2(n, dev, sig_map=0.25), False),
            (f"EKF3 + synthetic corner peaks (recall x{RECALL}, noise {NOISE}px)", lambda n: GateEKF3(n, dev, sig_map=0.25), True)]
for name, mk, up in variants:
    E, D, V = [], [], []
    for d in ds:
        e, di = run(mk, d, up)
        T, N = d["done"].shape
        es = torch.zeros_like(d["done"]); es[1:] = d["done"][:-1]; es[0] = True
        tt = torch.arange(T)[:, None].expand(T, N)
        st = torch.where(es, tt, torch.zeros_like(tt)).cummax(0).values
        E.append(e); D.append(di); V.append(d["valid"] & ((tt - st) * DT > 0.5))
    E, D, V = torch.cat(E, 1), torch.cat(D, 1), torch.cat(V, 1)
    print(name)
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15)):
        k = V & (D[:, :, 0] >= lo) & (D[:, :, 0] < hi)
        v = E[:, :, 0][k]
        print(f"   d {lo:>2}-{hi:<2} m: mean {v.mean():.2f} median {v.median():.2f} p90 {v.quantile(0.9):.2f}  (n {int(k.sum())})")
