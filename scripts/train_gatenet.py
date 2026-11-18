"""Train GateNet on Isaac-collected frames (isaac/collect_percep.py) with photometric / blur / noise augmentation.

Run: ~/isaac-env/bin/python -I scripts/train_gatenet.py --data runs/percep/d0/data.pt --epochs 30 --out runs/percep/g0
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.camera import augment  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402
from droneracing.gatenet import GateNet, decode, focal_loss, make_corner_targets, make_targets  # noqa: E402


def evaluate(net, data, idx, dev, H, W, bs=128):
    """Detection precision / recall and corner error on realistically augmented test frames, split by gate size."""
    net.eval()
    tp = fp = fn = 0
    err, ext_l = [], []
    gen = torch.Generator(device=dev)
    gen.manual_seed(0)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i : i + bs]
            img = realistic_augment(data["img"][j].to(dev).float() / 255.0, 1.0, gen=gen)
            px = data["corners"][j].to(dev).float() * torch.tensor([W, H], device=dev)
            heat, _, _, pos = make_targets(px, data["vis"][j].to(dev), data["slot_valid"][j].to(dev), H, W)
            dets = net.detect(img)
            for b, d in enumerate(dets):
                gtc = px[b][pos[b]]  # [g,4,2]
                used = set()
                for row in d:
                    c = row[1:].view(4, 2)
                    if len(gtc):
                        dist = (gtc - c).norm(dim=-1).mean(-1)
                        k = int(dist.argmin())
                        e_ = float((gtc[k].amax(0) - gtc[k].amin(0)).amax())
                        if dist[k] < max(12.0, 0.35 * e_) and k not in used:
                            used.add(k); tp += 1; err.append(float(dist[k])); ext_l.append(e_); continue
                    fp += 1
                fn += len(gtc) - len(used)
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    res = {"precision": p, "recall": r, "corner_err_px": sum(err) / max(1, len(err))}
    e, x = torch.tensor(err), torch.tensor(ext_l)
    for lo, hi in ((0, 30), (30, 60), (60, 1000)):
        m = (x >= lo) & (x < hi)
        if m.any():
            res[f"err_ext{lo}-{hi}"] = float(e[m].mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--max-per", type=int, default=0, help="frames kept per data file (0 = all)")
    ap.add_argument("--real-aug", type=float, default=0.8, help="fraction of each batch using realistic backgrounds / blur")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--out", default="runs/percep/g0")
    ap.add_argument("--refine", action="store_true", help="add corner heatmap / sub-cell offset heads")
    ap.add_argument("--init", default=None, help="warm start from a gatenet.pt (backbone + first 9 head channels)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    parts = []
    for p in a.data:
        x = torch.load(p)
        if a.max_per and len(x["img"]) > a.max_per:
            sel = torch.randperm(len(x["img"]), generator=torch.Generator().manual_seed(1))[: a.max_per]
            x = {k: (v[sel] if torch.is_tensor(v) and len(v) == len(x["img"]) else v) for k, v in x.items()}
        parts.append(x)
    data = {k: (torch.cat([x[k] for x in parts]) if torch.is_tensor(parts[0][k]) else parts[0][k]) for k in parts[0]}
    del parts
    H, W = data["hw"]
    n = len(data["img"])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    ntest = max(256, n // 10)
    te, tr = perm[:ntest], perm[ntest:]
    net = GateNet(refine=a.refine).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location=dev)["net"]
        mine = net.state_dict()
        for k_, v_ in sd.items():
            if k_ == "head.weight":
                mine[k_][: v_.shape[0]] = v_
            elif k_ == "head.bias":
                mine[k_][: v_.shape[0]] = v_
            else:
                mine[k_] = v_
        net.load_state_dict(mine)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        p = tr[torch.randperm(len(tr))]
        lh = lo = 0.0
        for i in range(len(tr) // a.bs):
            j = p[i * a.bs : (i + 1) * a.bs]
            img = data["img"][j].to(dev).float() / 255.0
            nr = int(len(j) * a.real_aug)
            img = torch.cat((realistic_augment(img[:nr], a.aug), augment(img[nr:], torch.zeros(len(j) - nr, 3, device=dev), a.aug)))
            px = data["corners"][j].to(dev).float() * torch.tensor([W, H], device=dev)
            heat, off_t, mask, _ = make_targets(px, data["vis"][j].to(dev), data["slot_valid"][j].to(dev), H, W)
            nout = net(img)
            logit, off = nout[0], nout[1]
            l_h = focal_loss(logit, heat)
            l_o = ((off - off_t).abs() * mask).sum() / mask.sum().clamp_min(1.0) / 8.0
            loss = l_h + 0.5 * l_o
            if a.refine:
                ch_t, co_t, cm = make_corner_targets(px, data["vis"][j].to(dev), data["slot_valid"][j].to(dev), H, W)
                l_c = focal_loss(nout[2], ch_t)
                l_co = ((nout[3] - co_t).abs() * cm).sum() / cm.sum().clamp_min(1.0)
                loss = loss + l_c + 2.0 * l_co
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            lh += l_h.item(); lo += l_o.item()
        nb = len(tr) // a.bs
        m = evaluate(net, data, te, dev, H, W) if ep % 3 == 0 or ep == a.epochs else {}
        print(f"ep {ep:3d} | heat {lh/nb:.3f} off {lo/nb:.2f} | {time.time()-t0:5.0f}s | " + " ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
        torch.save({"net": net.state_dict(), "hw": (H, W), "refine": a.refine}, out / "gatenet.pt")


if __name__ == "__main__":
    main()
