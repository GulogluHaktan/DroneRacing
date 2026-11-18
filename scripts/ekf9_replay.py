"""Drag-aided velocity (GateEKF9) vs GateEKF5 on recorded camera-line flights (eval_vision --dump; detections already delayed).
Drag coefficients per drone: least-squares fit of the body x / y specific force against -(|v| v_b, v_b) on the first 1000 steps
(ground truth, like a bench / flight-log calibration), then multiplied by (1 + N(0, ERR)) as calibration error.
Run: ~/isaac-env/bin/python -I scripts/ekf9_replay.py ERR runs/flight/g15_23/data.pt [...]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf5 import GateEKF5  # noqa: E402
from droneracing.ekf9 import GateEKF9  # noqa: E402
from droneracing.quad import quat_exp, quat_mul, quat_to_rot  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(4)
ERR = float(sys.argv[1])
H_, W_ = (616, 820) if "g15" in sys.argv[2] or "cal" in sys.argv[2] else (308, 410)
KW = dict(sig_map=0.25, kappa=0.3, kappa_corner=0.15, aided_perm=True, n_map=32, H=H_, W=W_, hfov=128.0, cam_tilt=0.7854)


def fit_drag(d):
    T, N = d["done"].shape
    R = quat_to_rot(d["gtq"].reshape(-1, 4)).reshape(T, N, 3, 3)
    vb = (R.transpose(-1, -2) @ d["gtv"][..., None])[..., 0]
    s = d["gtv"].norm(dim=-1)
    a = d["imu"][..., 3:] * 20.0
    cq, cl = torch.zeros(N), torch.zeros(N)
    for n in range(N):
        t = slice(0, min(1000, T))
        X = torch.cat((torch.stack((-(s[t, n] * vb[t, n, 0]), -vb[t, n, 0], torch.ones_like(s[t, n]), torch.zeros_like(s[t, n])), -1),
                       torch.stack((-(s[t, n] * vb[t, n, 1]), -vb[t, n, 1], torch.zeros_like(s[t, n]), torch.ones_like(s[t, n])), -1)))
        yv = torch.cat((a[t, n, 0], a[t, n, 1]))
        sol = torch.linalg.lstsq(X, yv[:, None]).solution[:, 0]
        cq[n], cl[n] = sol[0].clamp_min(0), sol[1].clamp_min(0)
    return cq, cl


def replay(d, drag, cq=None, cl=None):
    T, N = d["done"].shape
    ekf = GateEKF9(N, "cpu", sig_drag=float(__import__("os").environ.get("SIGD", "2.2")), **KW) if drag else GateEKF5(N, "cpu", **KW)
    if drag:
        ekf.set_drag(cq, cl)

    def reset(ids, t):
        q = quat_mul(d["gtq"][t][ids], quat_exp(torch.randn(len(ids), 3) * 0.02))
        ekf.reset(ids, d["gtp"][t][ids] + torch.randn(len(ids), 3) * 0.05, d["gtv"][t][ids], q)

    reset(torch.arange(N), 0)
    ev, ep = [], []
    for t in range(T):
        ekf.predict(d["imu"][t], 0.01)
        ekf.update(d["feat"][t], d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t], delay=5)
        ev.append((ekf.v - d["gtv"][t]).norm(dim=-1)); ep.append((ekf.p - d["gtp"][t]).norm(dim=-1))
        if d["done"][t].any() and t + 1 < T:
            reset(torch.nonzero(d["done"][t]).flatten(), t + 1)
    m = d["valid"]
    ev, ep = torch.stack(ev)[m], torch.stack(ep)[m]
    return f"vel err p50 {ev.median():.3f} mean {ev.mean():.3f} p90 {ev.quantile(.9):.3f} m/s | pos err p50 {ep.median():.2f} p90 {ep.quantile(.9):.2f} m"


for f in sys.argv[2:]:
    d = torch.load(f)
    cq, cl = fit_drag(d)
    print(f, "fitted c_quad", [round(x, 4) for x in cq.tolist()], "c_lin", [round(x, 3) for x in cl.tolist()])
    print("  GateEKF5           ", replay(d, False))
    g = torch.Generator().manual_seed(1)
    print(f"  GateEKF9 (err {ERR:.0%})", replay(d, True, cq * (1 + torch.randn(len(cq), generator=g) * ERR), cl * (1 + torch.randn(len(cl), generator=g) * ERR)))
