"""Train the GRU estimator on flight logs (analytic detector model: scripts/gen_estimator_data.py, or Isaac:
isaac/collect_flight.py). Windows are aligned with episode starts (the filter starts from a known pose),
long enough to cover most of a lap.

Run: ~/isaac-env/bin/python -I scripts/train_estimator.py --data runs_host/sim0/data.pt runs_host/sim1/data.pt --out runs_host/e1
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.estimator import Estimator, split_out  # noqa: E402

from droneracing.estimator import DET_DIM, K_DET, strapdown_sequence  # noqa: E402

NF = K_DET * DET_DIM  # detection features at the start of the input vector


def load(paths, dev):
    ds = [torch.load(p) for p in paths]
    d = {k: torch.cat([x[k] for x in ds], 1) for k in ds[0]}
    T, N = d["done"].shape
    done0 = d["done"]
    es = torch.zeros_like(done0)
    es[1:] = done0[:-1]
    es[0] = True
    xs = []
    for i in range(0, N, 2048):  # build in env chunks, keep half precision (GPU memory is tight)
        imu = d["imu"][:, i:i + 2048].float().to(dev)
        strap = strapdown_sequence(imu, es[:, i:i + 2048].to(dev))
        xs.append(torch.cat((d["feat"][:, i:i + 2048].float().to(dev), imu, d["act"][:, i:i + 2048].float().to(dev), strap), -1).half().cpu())
    x = torch.cat(xs, 1).to(dev)
    del xs
    lab = torch.cat((d["vel"], d["up"], d["gate"].reshape(T, N, 14)), -1).half().to(dev)
    done, valid = d["done"].to(dev), d["valid"].to(dev)
    return x, lab, done, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--L", type=int, default=384)
    ap.add_argument("--burn", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--init", default=None)
    ap.add_argument("--snap", type=float, default=1.0, help="fraction of windows aligned with an episode start")
    ap.add_argument("--out", default="runs_host/e1")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x, lab, done, valid = load(a.data, dev)
    T, N = done.shape
    ep_start = torch.zeros_like(done)
    ep_start[1:] = done[:-1]
    ep_start[0] = True
    tt = torch.arange(T, device=dev)[:, None].expand(T, N)
    start_t = torch.where(ep_start, tt, torch.zeros_like(tt)).cummax(0).values  # start of the episode containing t
    nval = max(2, N // 10)
    tr_envs, va_envs = torch.arange(0, N - nval, device=dev), torch.arange(N - nval, N, device=dev)
    ar = torch.arange(a.L, device=dev)

    def windows(envs, bs):
        e = envs[torch.randint(len(envs), (bs,), device=dev)]
        t0 = torch.randint(0, T - a.L + 1, (bs,), device=dev)
        snap = torch.rand(bs, device=dev) < a.snap
        t0 = torch.where(snap, start_t[t0, e].clamp(max=T - a.L), t0)
        return e, t0[:, None] + ar[None]

    net = Estimator(a.hidden).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)["net"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def run(e, ti, train):
        xb = x[ti, e[:, None]].float()  # [B,L,F]
        yb = lab[ti, e[:, None]].float()
        B = xb.shape[0]
        if train:  # sensor dropouts: dead frames and stale (repeated) frames (30 Hz camera on a 50 Hz loop)
            feat = xb[..., :NF]
            feat = torch.where(torch.rand(B, a.L, 1, device=dev) < 0.04, torch.zeros_like(feat), feat)
            stale = torch.rand(B, a.L, device=dev) < 0.3
            stale[:, 0] = False
            src = torch.where(~stale, ar[None].expand(B, -1), torch.zeros_like(ar)[None].expand(B, -1)).cummax(1).values
            xb = torch.cat((feat.gather(1, src[..., None].expand(-1, -1, NF)), xb[..., NF:]), -1)
        o, _ = net(xb)
        vel, up, rel, nrm, seen_l = split_out(o)
        g = yb[..., 6:].reshape(B, a.L, 2, 7)
        dn = done[ti, e[:, None]].long()
        alive = ((dn.cumsum(1) - dn) == 0).float()  # steps up to and including the episode end
        aligned = ep_start[ti[:, 0], e]
        burn = (ar[None] < a.burn) & ~aligned[:, None]
        m = alive * (~burn).float() * valid[ti, e[:, None]].float()
        sg = g[..., 6]
        l_v = (F.smooth_l1_loss(vel, yb[..., 0:3], reduction="none", beta=0.05).sum(-1) * m).sum() / m.sum()
        l_u = (((up - yb[..., 3:6]) ** 2).sum(-1) * m).sum() / m.sum()
        wg = sg * m[..., None]
        l_r = (F.smooth_l1_loss(rel, g[..., 0:3], reduction="none", beta=0.05).sum(-1) * wg).sum() / wg.sum().clamp_min(1)
        l_n = (((nrm - g[..., 3:6]) ** 2).sum(-1) * wg).sum() / wg.sum().clamp_min(1)
        l_s = (F.binary_cross_entropy_with_logits(seen_l, sg, reduction="none") * m[..., None]).sum() / (m.sum() * 2)
        loss = 10 * l_v + 30 * l_u + 4 * l_r + 2 * l_n + 0.5 * l_s
        with torch.no_grad():
            met = {
                "vel_err_m/s": float((((vel - yb[..., 0:3]) * 10).norm(dim=-1) * m).sum() / m.sum()),
                "up_err_deg": float((torch.acos((F.normalize(up, dim=-1) * yb[..., 3:6]).sum(-1).clamp(-1, 1)) * 57.3 * m).sum() / m.sum()),
                "gate0_err_m": float((((rel[..., 0, :] - g[:, :, 0, 0:3]) * 5).norm(dim=-1) * wg[..., 0]).sum() / wg[..., 0].sum().clamp_min(1)),
                "gate1_err_m": float((((rel[..., 1, :] - g[:, :, 1, 0:3]) * 5).norm(dim=-1) * wg[..., 1]).sum() / wg[..., 1].sum().clamp_min(1)),
                "seen_acc": float((((seen_l > 0).float() == sg).float() * m[..., None]).sum() / (m.sum() * 2)),
            }
        return loss, met

    t0 = time.time()
    for it in range(1, a.iters + 1):
        net.train()
        loss, _ = run(*windows(tr_envs, a.bs), True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 500 == 0 or it == a.iters:
            net.eval()
            with torch.no_grad():
                mets = [run(*windows(va_envs, 128), False)[1] for _ in range(6)]
            m = {k: sum(v[k] for v in mets) / len(mets) for k in mets[0]}
            print(f"it {it:5d} | loss {loss.item():.4f} | {time.time()-t0:4.0f}s | " + " ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
            torch.save({"net": net.state_dict(), "hidden": a.hidden}, out / "estimator.pt")


if __name__ == "__main__":
    main()
