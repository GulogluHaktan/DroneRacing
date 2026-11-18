"""Train the MonoRace GateNet (U-Net, f = 4, five outputs, Dice + 2 BCE deep supervision) on adaptive crops of the own camera and check QuAdGate on held-out crops.

Data: ``isaac/collect_percep.py --crop-out S`` (crops with inner-corner labels). Augmentation as in the paper where the labels allow it: random affine (rotation,
scale, translation) applied to image and corners together, HSV / colour / sensor noise / motion blur (``realistic_augment``), rolling shutter (rows shifted with the
labels). Schedule: AdamW 1e-3, learning rate x sqrt(0.1) at 10 %, 33 %, 66 %, 90 % of the epochs.
Run: ~/isaac-env/bin/python -I scripts/train_monorace_net.py --data runs/percep/d8/data.pt --epochs 40 --out runs_host/m0
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.gateseg import frame_masks  # noqa: E402
from droneracing.monorace import GateNetMR, all_corners_from_inner, deep_loss, predict_masks, quadgate  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402


def affine_batch(img, px, scale_rng=(0.8, 1.25), rot_deg=25.0, shift=0.12):
    """Random affine of img [B,3,S,S] and corner px [B,G,4,2] (pixel coords)."""
    B, _, S, _ = img.shape
    dev = img.device
    a = (torch.rand(B, device=dev) * 2 - 1) * math.radians(rot_deg)
    s = torch.exp(torch.empty(B, device=dev).uniform_(math.log(scale_rng[0]), math.log(scale_rng[1])))
    t = (torch.rand(B, 2, device=dev) * 2 - 1) * shift * S
    c = S / 2
    R = torch.stack((torch.stack((torch.cos(a), -torch.sin(a)), -1), torch.stack((torch.sin(a), torch.cos(a)), -1)), 1)  # [B,2,2]
    A = torch.zeros(B, 3, 3, device=dev)  # input px -> output px: p_out = s R (p_in - c) + c + t
    A[:, :2, :2] = s[:, None, None] * R
    A[:, :2, 2] = c + t - (s[:, None, None] * R @ torch.full((B, 2, 1), c, device=dev))[..., 0]
    A[:, 2, 2] = 1
    Ainv = torch.linalg.inv(A)
    N = torch.tensor([[2.0 / S, 0, -1], [0, 2.0 / S, -1], [0, 0, 1]], device=dev)
    theta = (N @ Ainv @ torch.linalg.inv(N))[:, :2]  # output normalized -> input normalized
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    ph = torch.cat((px, torch.ones_like(px[..., :1])), -1)  # [B,G,4,3]
    new = (A[:, None, None] @ ph[..., None])[..., 0][..., :2]
    return out, new


def rs_batch(img, px, f, max_ms):
    B, _, H, W = img.shape
    dev = img.device
    T = torch.rand(B, device=dev) * max_ms / 1000.0 * (torch.rand(B, device=dev) < 0.7).float()
    om = (torch.randn(B, 3, device=dev) * 3.5).clamp(-14, 14)
    tr = ((torch.arange(H, device=dev) + 0.5) / H - 0.5)[None] * T[:, None]
    du, dv = (f * -om[:, 1])[:, None] * tr, (f * om[:, 0])[:, None] * tr
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=img.dtype), torch.arange(W, device=dev, dtype=img.dtype), indexing="ij")
    grid = torch.stack((((xs[None] - du[:, :, None]) + 0.5) / W * 2 - 1, ((ys[None] - dv[:, :, None]) + 0.5) / H * 2 - 1), -1)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    t_r = (px[..., 1] / H - 0.5) * T[:, None, None]
    shift = torch.stack((f * -om[:, 1, None, None] * t_r, f * om[:, 0, None, None] * t_r), -1)
    return out, px + shift


@torch.no_grad()
def check_quadgate(net, data, idx, dev, S, n_max=300):
    net.eval()
    rng = np.random.default_rng(0)
    found = tot = 0
    errs, nm = [], []
    for i in idx.tolist():
        if tot >= n_max:
            break
        px = data["corners"][i].float() * S
        sv = data["slot_valid"][i]
        for s in range(px.shape[0]):
            if not bool(sv[s]):
                continue
            c = px[s]
            if not bool(((c[:, 0] >= 2) & (c[:, 0] < S - 2) & (c[:, 1] >= 2) & (c[:, 1] < S - 2)).all()):
                continue
            ext = float((c.amax(0) - c.amin(0)).amax())
            if ext < 20:
                continue
            c8 = all_corners_from_inner(c[None])[0]
            ok8 = (c8[:, 0] >= 0) & (c8[:, 0] < S) & (c8[:, 1] >= 0) & (c8[:, 1] < S)
            if int(ok8.sum()) < 4:
                continue
            up = c[0] - c[3]
            roll = math.atan2(float(up[0]), -float(up[1]))
            img = data["img"][i : i + 1].to(dev).float() / 255.0
            prob = predict_masks(net, img)[0]
            pri = c8.numpy() + rng.normal(size=(1, 2)) * 8 + rng.normal(size=(8, 2)) * 2
            out = quadgate(prob, pri[ok8.numpy()], np.zeros(int(ok8.sum()), int), np.nonzero(ok8.numpy())[0], roll)
            tot += 1
            if len(out) >= 3:
                found += 1
                nm.append(len(out) / int(ok8.sum()))
                errs.append(np.mean([np.linalg.norm(o[2:4] - c8[int(o[1])].numpy()) for o in out]))
    net.train()
    return found / max(tot, 1), (float(np.mean(nm)) if nm else 0.0), (float(np.median(errs)) if errs else float("nan")), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--f", type=int, default=4)
    ap.add_argument("--rs-ms-max", type=float, default=25.0)
    ap.add_argument("--init", default=None, help="checkpoint to continue from (weights only; use a lower --lr)")
    ap.add_argument("--out", default="runs_host/m0")
    a = ap.parse_args()
    dev = "cuda"
    parts = [torch.load(p, mmap=True) for p in a.data]
    data = {k: (torch.cat([x[k] for x in parts]) if torch.is_tensor(parts[0][k]) else parts[0][k]) for k in parts[0]}
    S = int(data["hw"][0])
    n = len(data["img"])
    f_px = (S / 2) / math.tan(math.radians(float(data["hfov"])) / 2)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    te, tr = perm[:1000], perm[1000:]
    net = GateNetMR(a.f).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)["net"])
    print(f"crops {n} ({S}x{S}), parameters {sum(p.numel() for p in net.parameters())/1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(tr) // a.bs
    miles = [int(a.epochs * x) * spe for x in (0.10, 0.33, 0.66, 0.90)]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, miles, gamma=math.sqrt(0.1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        sel = tr[torch.randperm(len(tr))]
        for i in range(spe):
            j = sel[i * a.bs : (i + 1) * a.bs].sort().values
            img = data["img"][j].to(dev).float() / 255.0
            px = data["corners"][j].to(dev).float() * S
            sv = data["slot_valid"][j].to(dev)
            img, px = affine_batch(img, px)
            if a.rs_ms_max > 0:
                img, px = rs_batch(img, px, f_px, a.rs_ms_max)
            img = realistic_augment(img, 1.0)
            tgt = frame_masks(px, sv, S, S)
            loss = deep_loss(net(img), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
        if ep % 5 == 0 or ep == a.epochs or ep == 1:
            fr, nmf, err, tot = check_quadgate(net, data, te, dev, S)
            print(f"ep {ep} | {time.time()-t0:5.0f}s | loss {float(loss):.3f} | QuAdGate on held-out gates: >=3 corners matched {fr*100:.0f} % of {tot}, share of corners {nmf*100:.0f} %, corner error median {err:.2f} px", flush=True)
        else:
            print(f"ep {ep} | {time.time()-t0:5.0f}s | loss {float(loss):.3f}", flush=True)
        torch.save({"net": net.state_dict(), "f": a.f, "S": S}, out / "monorace.pt")


if __name__ == "__main__":
    main()
