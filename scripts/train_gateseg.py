"""Train GateSeg (gate-frame segmentation, MonoRace GateNet style) on Isaac frames with inner-corner labels.

Mask labels are computed on the fly from the labelled inner corners (droneracing.gateseg.frame_masks). The test metric is the
inner-corner error of the edge-line extractor (droneracing.gateseg.extract_gates) against the labels, split into full gates
(4 corners in the image) and partial gates (1-3 corners in the image; the extractor extrapolates the rest).
Run: ~/isaac-env/bin/python -I scripts/train_gateseg.py --data runs/percep/d6/data.pt --epochs 12 --out runs_host/s0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac"))
from droneracing.camera import augment  # noqa: E402
from droneracing.gateseg import GateSeg, extract_gates, frame_masks  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402


def evaluate(net, data, idx, dev, H, W, bs=64):
    net.eval()
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    res = {"full": [], "partial": []}
    miss = {"full": 0, "partial": 0}
    iou_n = iou_d = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i : i + bs]
            img = realistic_augment(data["img"][j].to(dev).float() / 255.0, 1.0, gen=g)
            px = data["corners"][j].to(dev).float() * torch.tensor([W, H], device=dev)
            sv = data["slot_valid"][j].to(dev)
            tgt = frame_masks(px, sv, H, W)
            prob = torch.sigmoid(net(img))
            pm = prob > 0.5
            iou_n += float((pm & (tgt > 0.5)).sum()); iou_d += float((pm | (tgt > 0.5)).sum())
            pnp = prob[:, 0].cpu().numpy()
            pxc = px.cpu()
            cin = (pxc[..., 0] >= 0) & (pxc[..., 0] < W) & (pxc[..., 1] >= 0) & (pxc[..., 1] < H)
            for b in range(len(j)):
                dets = extract_gates(pnp[b] > 0.5, pnp[b])
                for s in range(px.shape[1]):
                    if not sv[b, s]:
                        continue
                    nin = int(cin[b, s].sum())
                    if nin == 0:
                        continue
                    ext = float((pxc[b, s].amax(0) - pxc[b, s].amin(0)).amax())
                    if ext < 12:  # tiny far gates are not the point here
                        continue
                    key = "full" if nin == 4 else "partial"
                    best = min((float((torch.tensor(inner, dtype=torch.float32) - pxc[b, s]).norm(dim=-1).mean()) for _, inner, _ in dets), default=1e9)
                    if best < 0.25 * ext + 5:
                        res[key].append(best)
                    else:
                        miss[key] += 1
    out = {"iou": iou_n / max(iou_d, 1)}
    for k in res:
        e = np.array(res[k]) if res[k] else np.array([np.nan])
        n = len(res[k]) + miss[k]
        out[f"{k}_found"] = len(res[k]) / max(n, 1)
        out[f"{k}_err_med"] = float(np.nanmedian(e))
        out[f"{k}_err_p90"] = float(np.nanpercentile(e, 90)) if res[k] else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--partial-boost", type=float, default=3.0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--rs-ms-max", type=float, default=0.0, help="train with a rolling shutter: readout time uniform in 0..this [ms] per sample, synthesised from the recorded body rate; corner labels are shifted with the image rows")
    ap.add_argument("--rs-prob", type=float, default=0.7)
    ap.add_argument("--tilt", type=float, default=0.785, help="camera up-tilt [rad] (for the body-to-camera rate)")
    ap.add_argument("--out", default="runs_host/s0")
    a = ap.parse_args()
    dev = "cuda"
    parts = [torch.load(p, mmap=True) for p in a.data]
    data = {k: (torch.cat([x[k] for x in parts]) if torch.is_tensor(parts[0][k]) else parts[0][k]) for k in parts[0]}
    H, W = data["hw"]
    n = len(data["img"])
    cor = data["corners"].float() * torch.tensor([W, H])
    cin = (cor[..., 0] >= 0) & (cor[..., 0] < W) & (cor[..., 1] >= 0) & (cor[..., 1] < H)
    partial = (data["slot_valid"] & (cin.sum(-1) >= 1) & (cin.sum(-1) < 4)).any(-1)
    print(f"frames {n} ({H}x{W}), with a partly visible gate {int(partial.sum())}", flush=True)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    ntest = max(512, n // 10)
    te, tr = perm[:ntest], perm[ntest:]
    wts = 1.0 + (a.partial_boost - 1.0) * partial[tr].float()
    net = GateSeg().to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)["net"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        sel = tr[torch.multinomial(wts, len(tr), replacement=True)]
        for i in range(spe):
            j = sel[i * a.bs : (i + 1) * a.bs].sort().values
            img = data["img"][j].to(dev).float() / 255.0
            nr = int(len(j) * 0.8)
            img = torch.cat((realistic_augment(img[:nr], 1.0), augment(img[nr:], torch.zeros(len(j) - nr, 3, device=dev), 1.0)))
            px = data["corners"][j].to(dev).float() * torch.tensor([W, H], device=dev)
            if a.rs_ms_max > 0:  # rolling shutter synthesised from the recorded body rate (same model as isaac/cam_artifacts.rolling_shutter)
                import math as _m
                ts_, tc_ = _m.sin(a.tilt), _m.cos(a.tilt)
                om_c = (torch.randn(len(j), 3, device=dev) * 3.5).clamp(-14, 14)  # the recorded omega is zero (frames were placed, not flown): sample race-like camera rates [rad/s]
                f_ = (W / 2) / _m.tan(_m.radians(float(data["hfov"])) / 2)
                T_ = torch.rand(len(j), device=dev) * a.rs_ms_max / 1000.0 * (torch.rand(len(j), device=dev) < a.rs_prob).float()
                tr_ = ((torch.arange(H, device=dev) + 0.5) / H - 0.5)[None] * T_[:, None]  # [B,H] row capture time
                du_, dv_ = (f_ * -om_c[:, 1])[:, None] * tr_, (f_ * om_c[:, 0])[:, None] * tr_
                ys_, xs_ = torch.meshgrid(torch.arange(H, device=dev, dtype=img.dtype), torch.arange(W, device=dev, dtype=img.dtype), indexing="ij")
                grid_ = torch.stack((((xs_[None] - du_[:, :, None]) + 0.5) / W * 2 - 1, ((ys_[None] - dv_[:, :, None]) + 0.5) / H * 2 - 1), -1)
                img = F.grid_sample(img, grid_, mode="bilinear", padding_mode="border", align_corners=False)
                t_r = (px[..., 1] / H - 0.5) * T_[:, None, None]
                px = px + torch.stack((f_ * -om_c[:, 1, None, None] * t_r, f_ * om_c[:, 0, None, None] * t_r), -1)
            tgt = frame_masks(px, data["slot_valid"][j].to(dev), H, W)
            logit = net(img)
            p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * tgt).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + tgt.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=torch.tensor(3.0, device=dev)) + dice.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
        m = evaluate(net, data, te[:1000], dev, H, W)
        print(f"ep {ep} | {time.time()-t0:5.0f}s | loss {float(loss):.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
        torch.save({"net": net.state_dict(), "hw": (H, W)}, out / "gateseg.pt")


if __name__ == "__main__":
    main()
