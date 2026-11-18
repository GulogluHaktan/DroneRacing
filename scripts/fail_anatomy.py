"""Per-episode outcome table from eval_vision --dump data (needs the finished / crashed / gates_passed fields).

For every episode that ends in a dump: drone index, mount error angle (--tilt-seed fixes the mount per drone index), outcome (FINISHED /
crashed / timeout), gates passed, and for crashes the gate index, altitude, speed, distance to the gate, true and believed in-plane miss and
the belief error (filter estimate vs true gate, body frame).
Run: ~/isaac-env/bin/python -I scripts/fail_anatomy.py runs/flight/f17_36/data.pt [...]
"""
import math
import sys

import torch

tot = {"FINISHED": 0, "crashed": 0, "timeout": 0}
for f in sys.argv[1:]:
    d = torch.load(f)
    if "finished" not in d:
        print(f, "has no finished/crashed fields: re-dump with the current eval_vision.py"); continue
    T, N = d["done"].shape
    ang = [math.degrees(math.acos(max(-1.0, min(1.0, (torch.trace(d["extr_err"][n]) - 1).item() / 2)))) for n in range(N)]
    print(f"== {f}  mount error angle per drone: " + " ".join(f"{a:.1f}" for a in ang))
    print("drone | ep | end_t | outcome  | gates | gate_idx | alt_m | speed | dist_gate | true_miss | bel_miss | belief_err")
    for n in range(N):
        for e, t in enumerate(torch.nonzero(d["done"][:, n]).flatten().tolist()):
            fin, cr = bool(d["finished"][t, n]), bool(d["crashed"][t, n])
            out = "FINISHED" if fin else ("crashed" if cr else "timeout")
            tot[out] += 1
            if fin:
                print(f"{n:5d} | {e:2d} | {t:5d} | {out:8s} | {int(d['gates_passed'][t, n]):5d} |"); continue
            sl = slice(max(0, t - 3), t + 1)
            pos = d["gtp"][t, n]; vel = d["gtv"][t, n]
            tr = d["gate"][sl, n, 0, 0:3].mean(0) * 5.0
            es = d["est"][sl, n, 6:9].mean(0) * 5.0
            print(f"{n:5d} | {e:2d} | {t:5d} | {out:8s} | {int(d['gates_passed'][t, n]):5d} | {int(d['gi'][t, n]):8d} | {pos[2].item():5.2f} | {vel.norm().item():5.1f} | "
                  f"{tr.norm().item():9.2f} | {tr[1:].norm().item():9.2f} | {es[1:].norm().item():8.2f} | {(tr - es).norm().item():10.2f}")
print("episodes:", tot)
