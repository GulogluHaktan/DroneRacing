"""Train the opponent head on the frozen GateNet trunk (gate outputs stay identical).
Run: ~/isaac-env/bin/python -I scripts/train_opphead.py --data runs/percep/o1/data.pt --gatenet runs_host/g17/gatenet.pt --out runs_host/g17o
Data: isaac/collect_opp.py (crop 384, keys opp_c / opp_r / opp_valid / opp_dist)."""
import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.camera import augment  # noqa: E402
from droneracing.derotate import derotate_img, rotate_points, roll_from_quat  # noqa: E402
from droneracing.gatenet import GateNet, focal_loss  # noqa: E402
from droneracing.oppnet import OppHead, decode, make_targets  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402

BINS = ((0, 4), (4, 8), (8, 12), (12, 20))


def batch(data, j, dev, H, W, cam_tilt, noise_deg, gen=None, train=True):
    img = data["img"][j].to(dev).float() / 255.0
    img = realistic_augment(img, 1.0, gen=gen) if gen is not None or train else img
    c = data["opp_c"][j].to(dev).float() * torch.tensor([W, H], device=dev)
    r = data["opp_r"][j].to(dev).float() * W
    v = data["opp_valid"][j].to(dev)
    q = data["state"][j][:, 6:10].to(dev).float()
    roll = roll_from_quat(q, cam_tilt) + torch.randn(len(j), generator=gen, device=dev) * math.radians(noise_deg)
    img = derotate_img(img, roll)
    c = rotate_points(c, -roll, W / 2, H / 2)
    return img, c, r, v


@torch.no_grad()
def evaluate(net, head, data, idx, dev, H, W, cam_tilt, thr=0.3, bs=64):
    net.eval(); head.eval()
    g = torch.Generator(device=dev); g.manual_seed(0)
    hit = torch.zeros(len(BINS)); tot = torch.zeros(len(BINS)); rerr = []; tp = fp = 0
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        img, c, r, v = batch(data, j, dev, H, W, cam_tilt, 3.0, g, False)
        net(img)
        dets = decode(head(net._y), thr)
        dist = data["opp_dist"][j].to(dev).float()
        for b, d in enumerate(dets):
            used = torch.zeros(len(d), dtype=torch.bool, device=dev)
            for m in range(v.shape[1]):
                if not v[b, m]:
                    continue
                k = next(k for k, (lo, hi) in enumerate(BINS) if lo <= dist[b, m] < hi) if dist[b, m] < BINS[-1][1] else None
                if k is None:
                    continue
                tot[k] += 1
                if len(d):
                    e = (d[:, 1:3] - c[b, m]).norm(dim=-1)
                    a = int(e.argmin())
                    if e[a] < max(4.0, float(r[b, m])) and not used[a]:
                        hit[k] += 1; used[a] = True; rerr.append(float(d[a, 3] / r[b, m]))
            tp += int(used.sum()); fp += int((~used).sum())
    out = {f"recall_{lo}-{hi}m": float(hit[k] / max(tot[k], 1)) for k, (lo, hi) in enumerate(BINS)}
    out["precision"] = tp / max(1, tp + fp); out["fp_per_frame"] = fp / max(1, len(idx))
    out["radius_ratio_med"] = float(torch.tensor(rerr).median()) if rerr else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--gatenet", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--cam-tilt", type=float, default=0.7854)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    parts = [torch.load(p) for p in a.data]
    data = {k: (torch.cat([x[k] for x in parts]) if torch.is_tensor(parts[0][k]) else parts[0][k]) for k in parts[0]}
    H, W = data["hw"]
    n = len(data["img"])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    te, tr = perm[: max(512, n // 10)], perm[max(512, n // 10):]
    ck = torch.load(a.gatenet, map_location=dev)
    net = GateNet(refine=ck.get("refine", False)).to(dev)
    net.load_state_dict(ck["net"]); net.eval()
    for p_ in net.parameters():
        p_.requires_grad_(False)
    head = OppHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"frames {n}, with opponent {int(data['opp_valid'].any(-1).sum())}", flush=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        head.train()
        sel = tr[torch.randperm(len(tr))]
        for i in range(spe):
            j = sel[i * a.bs:(i + 1) * a.bs]
            img, c, r, v = batch(data, j, dev, H, W, a.cam_tilt, 3.0)
            with torch.no_grad():
                net(img)
                y = net._y
            o = head(y)
            heat, reg, mask = make_targets(c, r, v, H, W)
            loss = focal_loss(o[:, :1], heat) + 2.0 * ((o[:, 1:] - reg).abs() * mask).sum() / mask.sum().clamp_min(1.0) / 3.0
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        m = evaluate(net, head, data, te[:1500], dev, H, W, a.cam_tilt)
        print(f"ep {ep} | {time.time()-t0:5.0f}s | loss {float(loss):.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
        torch.save({"head": head.state_dict(), "gatenet": a.gatenet, "hw": (H, W)}, out / "opphead.pt")


if __name__ == "__main__":
    main()
