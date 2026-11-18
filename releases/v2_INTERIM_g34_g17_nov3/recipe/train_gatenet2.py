"""Fine-tune GateNet (refine) so that corners of partly visible gates are found even when the gate centre is outside the image.

Run: ~/isaac-env/bin/python -I scripts/train_gatenet2.py --data runs/percep/d1/data.pt runs/percep/d2/data.pt --init runs_host/g3/gatenet.pt --out runs_host/g5
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.camera import augment  # noqa: E402
from droneracing.gatenet import GateNet, decode_refined, focal_loss, make_corner_targets, make_targets  # noqa: E402
from droneracing.gatenet2 import decode_corner_peaks, make_corner_targets_any  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402


def derot_batch(img, px, data, j, cam_tilt, noise_deg, gen=None):
    """Rotate the input by -roll (camera roll from the recorded attitude + a filter-like error) so world up is image up (MonoRace)."""
    from droneracing.derotate import derotate_img, derotate_labels, roll_from_quat
    q = data["state"][j][:, 6:10].to(img.device).float()
    roll = roll_from_quat(q, cam_tilt) + torch.randn(len(j), generator=gen, device=img.device) * math.radians(noise_deg)
    return derotate_img(img, roll), derotate_labels(px, roll, img.shape[2], img.shape[3])


def evaluate(net, data, idx, dev, H, W, bs=32, rc=False, derot=None):
    """Corner recall / pixel error of the corner-peak decode by number of corners in the image, and centre-detection recall.
    rc: test frames with the real-camera effects (droneracing/realcam_aug.py, fixed seed), labels moved with the image."""
    net.eval()
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    hit, tot, err = torch.zeros(5), torch.zeros(5), [[] for _ in range(5)]
    fp = npk = 0
    g_tot = g_hit = 0  # gates with >= 2 labelled corners in the image; found = >= 2 of them within 4 px (enough for a corner update)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i : i + bs]
            img = realistic_augment(data["img"][j].to(dev).float() / 255.0, 1.0, gen=g)
            px = data["corners"][j].to(dev).float() * torch.tensor([W, H], device=dev)
            vis_, sv_ = data["vis"][j].to(dev), data["slot_valid"][j].to(dev)
            if rc:
                from droneracing.realcam_aug import realcam
                img, px, vis_, sv_ = realcam(img, px, vis_, sv_, float(data["hfov"]), gen=g, win=data["win"][j] if "win" in data else None, full_hw=data.get("full_hw"))
            if derot is not None:  # (cam_tilt, roll error deg)
                img, px = derot_batch(img, px, data, j, derot[0], derot[1], g)
            _, _, _, pos = make_corner_targets_any(px, vis_, sv_, H, W)
            o = net(img)
            peaks = decode_corner_peaks(o[2], o[3], thr=0.3)
            cin = (px[..., 0] >= 0) & (px[..., 0] < W) & (px[..., 1] >= 0) & (px[..., 1] < H)
            nin = cin.sum(-1)
            for b, pk in enumerate(peaks):
                npk += len(pk)
                used = torch.zeros(len(pk), dtype=torch.bool, device=dev)
                for s in range(px.shape[1]):
                    n_pos = int(pos[b, s].sum()); n_found = 0
                    for c in range(4):
                        if not pos[b, s, c]:
                            continue
                        k = int(nin[b, s]); tot[k] += 1
                        m = (pk[:, 0] == c)
                        if m.any():
                            d = (pk[m][:, 2:] - px[b, s, c]).norm(dim=-1)
                            if d.min() < 4.0:
                                hit[k] += 1; err[k].append(float(d.min())); used[torch.nonzero(m).flatten()[d.argmin()]] = True; n_found += 1
                    if n_pos >= 2:
                        g_tot += 1; g_hit += int(n_found >= 2)
                fp += int((~used).sum())
    res = {f"recall_{k}in": float(hit[k] / max(tot[k], 1)) for k in range(1, 5)}
    res.update({f"err_{k}in": (sum(err[k]) / max(1, len(err[k]))) for k in range(1, 5)})
    res["unmatched_peaks_per_frame"] = fp / max(1, len(idx))
    res["gate_found_2c"] = g_hit / max(1, g_tot)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--init", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-per", type=int, default=0)
    ap.add_argument("--partial-boost", type=float, default=3.0, help="sampling weight of frames containing a partly visible gate")
    ap.add_argument("--rs-ms-max", type=float, default=0.0, help="rolling shutter readout time [ms] sampled uniformly up to this value (same model as isaac/cam_artifacts.rolling_shutter, corner labels shifted with their row)")
    ap.add_argument("--realcam", action="store_true", help="real-camera effects with consistent labels (droneracing/realcam_aug.py): rolling shutter from rotation and forward motion, physical motion blur, short exposure, lens edge, corrupt bands; replaces --rs-ms-max")
    ap.add_argument("--derotate", type=float, default=-1.0, help=">= 0: derotate the input with the camera roll (MonoRace); value = std of the roll error [deg] (filter attitude)")
    ap.add_argument("--cam-tilt", type=float, default=0.7854, help="camera tilt of the data set [rad] (own camera 45 deg)")
    ap.add_argument("--rs-prob", type=float, default=0.7, help="fraction of samples with a rolling shutter")
    ap.add_argument("--holdout", action="store_true", help="test split = the per-file test frames of scripts/percep_breakdown.py (randperm seed 0 of each file); keeps several data files held out")
    ap.add_argument("--out", default="runs_host/g5")
    a = ap.parse_args()
    dev = "cuda"
    parts = []
    for p in a.data:
        x = torch.load(p, mmap=True)  # several large data files: concatenation is the only full copy in memory
        if a.max_per and len(x["img"]) > a.max_per:
            sel = torch.randperm(len(x["img"]), generator=torch.Generator().manual_seed(1))[: a.max_per]
            x = {k: (v[sel] if torch.is_tensor(v) and len(v) == len(x["img"]) else v) for k, v in x.items()}
        parts.append(x)
    sizes = [len(x["img"]) for x in parts]
    data = {k: (torch.cat([x[k] for x in parts]) if torch.is_tensor(parts[0][k]) else parts[0][k]) for k in parts[0]}
    del parts
    H, W = data["hw"]
    n = len(data["img"])
    cor = data["corners"].float() * torch.tensor([W, H])
    cin = (cor[..., 0] >= 0) & (cor[..., 0] < W) & (cor[..., 1] >= 0) & (cor[..., 1] < H)
    partial = (data["slot_valid"] & (cin.sum(-1) >= 1) & (cin.sum(-1) < 4)).any(-1)
    print(f"frames {n}, with a partly visible gate {int(partial.sum())}", flush=True)
    if a.holdout:
        held, off = [], 0
        for m in sizes:
            held.append(off + torch.randperm(m, generator=torch.Generator().manual_seed(0))[: max(512, m // 10)])
            off += m
        te = torch.cat(held)
        mask = torch.ones(n, dtype=torch.bool)
        mask[te] = False
        tr = torch.nonzero(mask).flatten()
        tr = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(0))]
        te = te[torch.randperm(len(te), generator=torch.Generator().manual_seed(0))]
    else:
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        ntest = max(512, n // 10)
        te, tr = perm[:ntest], perm[ntest:]
    wts = 1.0 + (a.partial_boost - 1.0) * partial[tr].float()
    ck = torch.load(a.init, map_location=dev)
    net = GateNet(refine=True).to(dev)
    net.load_state_dict(ck["net"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    spe = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    DR = (a.cam_tilt, 3.0) if a.derotate >= 0 else None
    print("before:", {k: round(v, 3) for k, v in evaluate(net, data, te[:1500], dev, H, W, derot=DR).items()}, flush=True)
    print("before, real-camera test:", {k: round(v, 3) for k, v in evaluate(net, data, te[:1500], dev, H, W, rc=True, derot=DR).items()}, flush=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train()
        sel = tr[torch.multinomial(wts, len(tr), replacement=True)]
        for i in range(spe):
            j = sel[i * a.bs : (i + 1) * a.bs]
            img = data["img"][j].to(dev).float() / 255.0
            nr = int(len(j) * 0.8)
            img = torch.cat((realistic_augment(img[:nr], 1.0), augment(img[nr:], torch.zeros(len(j) - nr, 3, device=dev), 1.0)))
            px = data["corners"][j].to(dev).float() * torch.tensor([W, H], device=dev)
            vis, sv = data["vis"][j].to(dev), data["slot_valid"][j].to(dev)
            if a.realcam:
                from droneracing.realcam_aug import realcam
                img, px, vis, sv = realcam(img, px, vis, sv, float(data["hfov"]), rs_ms_max=max(a.rs_ms_max, 30.0), win=data["win"][j] if "win" in data else None, full_hw=data.get("full_hw"))
            elif a.rs_ms_max > 0:  # rolling shutter; the recorded body rate is zero (frames were placed): sample race-like camera rates [rad/s]
                import math as _m
                import torch.nn.functional as F
                om_c = (torch.randn(len(j), 3, device=dev) * 3.5).clamp(-14, 14)
                f_ = (W / 2) / _m.tan(_m.radians(float(data["hfov"])) / 2)
                T_ = torch.rand(len(j), device=dev) * a.rs_ms_max / 1000.0 * (torch.rand(len(j), device=dev) < a.rs_prob).float()
                tr_ = ((torch.arange(H, device=dev) + 0.5) / H - 0.5)[None] * T_[:, None]
                du_, dv_ = (f_ * -om_c[:, 1])[:, None] * tr_, (f_ * om_c[:, 0])[:, None] * tr_
                ys_, xs_ = torch.meshgrid(torch.arange(H, device=dev, dtype=img.dtype), torch.arange(W, device=dev, dtype=img.dtype), indexing="ij")
                grid_ = torch.stack((((xs_[None] - du_[:, :, None]) + 0.5) / W * 2 - 1, ((ys_[None] - dv_[:, :, None]) + 0.5) / H * 2 - 1), -1)
                img = F.grid_sample(img, grid_, mode="bilinear", padding_mode="border", align_corners=False)
                t_r = (px[..., 1] / H - 0.5) * T_[:, None, None]
                px = px + torch.stack((f_ * -om_c[:, 1, None, None] * t_r, f_ * om_c[:, 0, None, None] * t_r), -1)
            if a.derotate >= 0:
                img, px = derot_batch(img, px, data, j, a.cam_tilt, a.derotate)
            heat, off_t, mask, _ = make_targets(px, vis, sv, H, W)
            o = net(img)
            loss = focal_loss(o[0], heat) + 0.5 * ((o[1] - off_t).abs() * mask).sum() / mask.sum().clamp_min(1.0) / 8.0
            ch_t, co_t, cm, _ = make_corner_targets_any(px, vis, sv, H, W)
            loss = loss + focal_loss(o[2], ch_t) + 2.0 * ((o[3] - co_t).abs() * cm).sum() / cm.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
        m = evaluate(net, data, te[:1500], dev, H, W, derot=DR)
        print(f"ep {ep} | {time.time()-t0:5.0f}s | " + " ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
        mr = evaluate(net, data, te[:1500], dev, H, W, rc=True, derot=DR)
        print(f"   real-camera test | " + " ".join(f"{k} {v:.3f}" for k, v in mr.items()), flush=True)
        torch.save({"net": net.state_dict(), "hw": (H, W), "refine": True}, out / "gatenet.pt")


if __name__ == "__main__":
    main()
