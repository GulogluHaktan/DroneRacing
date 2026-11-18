"""Gate-only detector vs detector with an opponent head, on the SAME synthetic frames (NOT RUN YET: the opponent head and the data do not exist).

Question: does adding the opponent head (a) keep the gate detection of g17, (b) find the opponents, (c) stay as fast?
usage: python scripts/compare_opp_detector.py --data runs/percep/o_test/data.pt --gate-only runs_host/g17/gatenet.pt --with-opp runs_host/g17o/gatenet.pt [--ratios 0.1 0.3 0.5]

Data contract (extends the files of isaac/collect_percep.py; same keys as train_gatenet2 plus):
  data["opp_box"]   [N, M, 4] float, (x0, y0, x1, y1) of each opponent body in 0..1 image coordinates (same crop / derotation as "corners")
  data["opp_valid"] [N, M] bool, the opponent is inside the image (and larger than 4 px)
  data["opp_dist"]  [N, M] float, distance camera -> opponent [m]
Opponent head contract (to implement together with the head): net(img) returns the gate outputs o[0..3] as before and
  o[4] = opponent centre logits, o[5] = (dx, dy, log w, log h) offsets; decode_opp(o) below turns them into boxes.

Reported per model, for three frame subsets (all / opponent in view / no opponent):
  gate_found_2c, recall_{1..4}in, err_{k}in (pixel), unmatched_peaks_per_frame      -> gate detection (train_gatenet2.evaluate)
  gate peaks inside an opponent box per frame                                        -> the opponent drone is mistaken for a gate corner
  opponent recall / precision by distance bin (with-opp model only)
  mixture at the given opponent-in-view ratios: r * (in view) + (1 - r) * (no opponent)
  forward time per frame (batch 1, fp32 and fp16, ms)
Pass rule (printed at the end): gate_found_2c of the with-opp model >= gate-only - 0.01 on all subsets, opponent recall >= 0.90 for dist < 15 m,
precision >= 0.90, forward time <= 1.10 x gate-only.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from droneracing.gatenet import GateNet  # noqa: E402
from droneracing.gatenet2 import decode_corner_peaks  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402
from train_gatenet2 import evaluate as gate_eval  # noqa: E402

DIST_BINS = ((0, 5), (5, 10), (10, 15), (15, 25))


def load_net(path, dev):
    ck = torch.load(path, map_location=dev)
    net = GateNet().to(dev)
    sd = ck["net"] if isinstance(ck, dict) and "net" in ck else ck
    net.load_state_dict(sd, strict=False)  # the with-opp model has extra head weights; the gate-only model loads strictly the same keys
    return net.eval()


def decode_opp(o, thr=0.4, stride=2, k=4):
    """Opponent boxes [x0, y0, x1, y1, score] in pixels per frame. TO IMPLEMENT with the head (centre heatmap + size regression)."""
    raise NotImplementedError("opponent head not implemented yet")


def iou(a, b):
    x0, y0 = torch.maximum(a[0], b[0]), torch.maximum(a[1], b[1])
    x1, y1 = torch.minimum(a[2], b[2]), torch.minimum(a[3], b[3])
    inter = (x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter).clamp_min(1e-6))


@torch.no_grad()
def peaks_in_opp_box(net, data, idx, dev, H, W, bs=32):
    """Mean number of gate-corner peaks per frame that lie inside an opponent box (false gate corners on the opponent)."""
    g = torch.Generator(device=dev); g.manual_seed(0)
    n = 0
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        img = realistic_augment(data["img"][j].to(dev).float() / 255.0, 1.0, gen=g)
        box = data["opp_box"][j].to(dev).float() * torch.tensor([W, H, W, H], device=dev)
        val = data["opp_valid"][j].to(dev)
        for b, pk in enumerate(decode_corner_peaks(*(lambda o: (o[2], o[3]))(net(img)), thr=0.3)):
            for m in range(box.shape[1]):
                if val[b, m]:
                    x, y = pk[:, 2], pk[:, 3]
                    n += int(((x >= box[b, m, 0]) & (x <= box[b, m, 2]) & (y >= box[b, m, 1]) & (y <= box[b, m, 3])).sum())
    return n / max(1, len(idx))


@torch.no_grad()
def opp_metrics(net, data, idx, dev, H, W, bs=32, iou_thr=0.3):
    g = torch.Generator(device=dev); g.manual_seed(0)
    tp = {b: 0 for b in DIST_BINS}; tot = {b: 0 for b in DIST_BINS}; npred = ntp = 0
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        img = realistic_augment(data["img"][j].to(dev).float() / 255.0, 1.0, gen=g)
        gt = data["opp_box"][j].to(dev).float() * torch.tensor([W, H, W, H], device=dev)
        val, dist = data["opp_valid"][j].to(dev), data["opp_dist"][j].to(dev)
        preds = decode_opp(net(img))
        for b in range(len(j)):
            pb = preds[b]
            npred += len(pb)
            hit_p = torch.zeros(len(pb), dtype=torch.bool)
            for m in range(gt.shape[1]):
                if not val[b, m]:
                    continue
                bin_ = next((bb for bb in DIST_BINS if bb[0] <= float(dist[b, m]) < bb[1]), None)
                if bin_ is None:
                    continue
                tot[bin_] += 1
                for p in range(len(pb)):
                    if iou(pb[p, :4], gt[b, m]) >= iou_thr:
                        tp[bin_] += 1; hit_p[p] = True
                        break
            ntp += int(hit_p.sum())
    rec = {f"{b[0]}-{b[1]}m": tp[b] / max(1, tot[b]) for b in DIST_BINS}
    return {"recall": rec, "precision": ntp / max(1, npred)}


@torch.no_grad()
def forward_ms(net, dev, H, W, n=200, half=False):
    x = torch.rand(1, 3, H, W, device=dev)
    net = net.half() if half else net.float()
    x = x.half() if half else x
    for _ in range(20):
        net(x)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        net(x)
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gate-only", required=True)
    ap.add_argument("--with-opp", required=True)
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.3, 0.5], help="share of frames with an opponent in view; replace with the value measured in the simulator (scripts/measure_opp_ratio.py)")
    ap.add_argument("--max-frames", type=int, default=6000)
    ap.add_argument("--realcam", action="store_true", help="evaluate with the real-camera effects (as the g17 evaluation)")
    a = ap.parse_args()
    dev = "cuda"
    data = torch.load(a.data, mmap=True)
    H, W = data["hw"]
    n = len(data["img"])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))[: a.max_frames]
    inview = data["opp_valid"].any(-1)
    subsets = {"all": perm, "opponent in view": perm[inview[perm]], "no opponent": perm[~inview[perm]]}
    print(f"frames {len(perm)}; with an opponent in view {len(subsets['opponent in view'])} ({100*len(subsets['opponent in view'])/len(perm):.1f} %)")
    nets = {"gate-only": load_net(a.gate_only, dev), "with-opp": load_net(a.with_opp, dev)}
    res = {}
    for name, net in nets.items():
        res[name] = {}
        for sname, idx in subsets.items():
            r = gate_eval(net, data, idx, dev, H, W, rc=a.realcam)
            r["peaks_in_opp_box"] = peaks_in_opp_box(net, data, idx, dev, H, W) if sname != "no opponent" else 0.0
            res[name][sname] = r
            print(f"{name:10s} | {sname:17s} | gate_found_2c {r['gate_found_2c']:.3f} | recall4 {r['recall_4in']:.3f} | err4 {r['err_4in']:.2f} px | unmatched {r['unmatched_peaks_per_frame']:.2f}/frame | peaks in opponent box {r['peaks_in_opp_box']:.2f}/frame")
    om = opp_metrics(nets["with-opp"], data, subsets["opponent in view"], dev, H, W)
    print("opponent recall by distance:", {k: round(v, 3) for k, v in om["recall"].items()}, "| precision", round(om["precision"], 3))
    for r_ in a.ratios:
        for name in nets:
            mix = r_ * res[name]["opponent in view"]["gate_found_2c"] + (1 - r_) * res[name]["no opponent"]["gate_found_2c"]
            print(f"mixture, opponent in view {r_:.2f}: {name:10s} gate_found_2c {mix:.3f}")
    t = {nm: (forward_ms(load_net(p, dev), dev, H, W), forward_ms(load_net(p, dev), dev, H, W, half=True)) for nm, p in (("gate-only", a.gate_only), ("with-opp", a.with_opp))}
    print("forward ms (fp32 / fp16):", {k: tuple(round(x, 2) for x in v) for k, v in t.items()})
    ok_gate = all(res["with-opp"][s]["gate_found_2c"] >= res["gate-only"][s]["gate_found_2c"] - 0.01 for s in subsets if len(subsets[s]))
    ok_rec = all(om["recall"][f"{b[0]}-{b[1]}m"] >= 0.90 for b in DIST_BINS[:3])
    ok_prec = om["precision"] >= 0.90
    ok_time = t["with-opp"][0] <= 1.10 * t["gate-only"][0]
    print(f"PASS gate {ok_gate} | opponent recall<15m {ok_rec} | precision {ok_prec} | time {ok_time}  ->  {'ADOPT' if ok_gate and ok_rec and ok_prec and ok_time else 'DO NOT ADOPT'}")


if __name__ == "__main__":
    main()
