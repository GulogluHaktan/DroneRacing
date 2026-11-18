"""CPU/GPU-free test of GateEKF8 (MonoRace corner measurements with identity) on recorded camera-line flights (eval_vision --dump).

Synthetic QuAdGate output: the 8 corners (4 inner + 4 outer) of the next gates are projected with the ground-truth pose into the camera,
the true gate position is the map position plus a fixed random offset per gate (sigma 0.25 m, the map error of the filter). A frame gives
corners only if at least 3 are inside the image (QuAdGate min_match); pixel noise ``sig`` and a frame dropout are added. The measurements
reach the filter ``D`` steps late (``delay=D``) at the camera rate.
Compared: GateEKF5 with the recorded GateNet detections (reference line) and GateEKF8 with the synthetic corners.
Metric: position error of the filter (p90 by distance to the next gate) and error of the corrected next-gate position.
Run: ~/isaac-env/bin/python -I scripts/ekf8_replay.py 5 [sig_px] [dropout] runs/flight/own_41/data.pt [...]
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.ekf5 import GateEKF5  # noqa: E402
from droneracing.ekf8 import GateEKF8  # noqa: E402
from droneracing.quad import quat_exp, quat_mul  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(4)
D = int(sys.argv[1])
SIG = float(sys.argv[2])
DROP = float(sys.argv[3])
files = sys.argv[4:]
DUMP_DELAYED = bool(int(__import__("os").environ.get("DUMP_DELAYED", "0")))
dev, DT, PER = "cpu", 0.01, 3  # 30 Hz camera at 100 Hz
H, W, HFOV, TILT = 308, 410, 128.0, 0.7854
KW = dict(sig_map=0.25, kappa=0.3, kappa_corner=0.15, aided_perm=True, n_map=32, H=H, W=W, hfov=HFOV, cam_tilt=TILT)


def synth(ekf8, d, t, offs, J=2):
    """Synthetic corner list [N,16,5] seen at step t (capture time) from the ground-truth pose."""
    N = d["done"].shape[1]
    peaks = torch.zeros(N, 16, 5)
    if t % PER != 0 or t < 0:
        return peaks
    gi, mc, mn = d["gi"][t], d["mc"][t], d["mn"][t]
    from droneracing.quad import quat_to_rot

    R = quat_to_rot(d["gtq"][t])
    ar = torch.arange(N)
    cnt = torch.zeros(N, dtype=torch.long)
    for j in range(J):
        c = mc[:, j] + offs[ar, (gi + j).clamp(max=offs.shape[1] - 1)]
        uvs, oks = [], []
        for k in range(8):
            Pw = ekf8._corner_world(c, mn[:, j], torch.full((N,), k, dtype=torch.long))
            Pb = (R.transpose(1, 2) @ (Pw - d["gtp"][t])[..., None])[..., 0]
            Xc = (Pb - ekf8.t_cam) @ ekf8.Rb
            Z = Xc[:, 2].clamp_min(0.3)
            uv = torch.stack((ekf8.f * Xc[:, 0] / Z + W / 2, ekf8.f * Xc[:, 1] / Z + H / 2), -1)
            ok = (Xc[:, 2] > 0.3) & (Pb.norm(dim=-1) < 25) & (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)
            uvs.append(uv + torch.randn_like(uv) * SIG); oks.append(ok)
        uvs, oks = torch.stack(uvs, 1), torch.stack(oks, 1)  # [N,8,2], [N,8]
        for k in range(8):
            m = oks[:, k]
            peaks[ar, cnt.clamp(max=15), 0] = torch.where(m, (gi + j).float(), peaks[ar, cnt.clamp(max=15), 0])
            peaks[ar, cnt.clamp(max=15), 1] = torch.where(m, torch.full((N,), float(k)), peaks[ar, cnt.clamp(max=15), 1])
            peaks[ar, cnt.clamp(max=15), 2:4] = torch.where(m[:, None], uvs[:, k], peaks[ar, cnt.clamp(max=15), 2:4])
            peaks[ar, cnt.clamp(max=15), 4] = m.float()
            cnt = cnt + m.long()
    keep = (cnt >= 3) & (torch.rand(N) > DROP)
    peaks[~keep] = 0
    return peaks


def replay(mode, d, offs):
    T, N = d["done"].shape
    ekf = GateEKF8(N, dev, **KW) if mode == "mr" else GateEKF5(N, dev, **KW)
    zero = torch.zeros_like(d["feat"][0])
    ref8 = GateEKF8(N, dev, **KW)  # camera model only

    def reset(ids, t):
        q = quat_mul(d["gtq"][t][ids], quat_exp(torch.randn(len(ids), 3) * 0.02))
        ekf.reset(ids, d["gtp"][t][ids] + torch.randn(len(ids), 3) * 0.05, d["gtv"][t][ids], q)

    reset(torch.arange(N), 0)
    perr, gerr, dist, nmr = [], [], [], []
    ar = torch.arange(N)
    for t in range(T):
        ekf.predict(d["imu"][t], DT)
        td = t - D
        if mode == "mr":
            pk = synth(ref8, d, td, offs) if td >= 0 else torch.zeros(N, 16, 5)
            ekf.update(zero, d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t], peaks=pk, delay=D)
            nmr.append(ekf.mr_updates.clone())
        else:
            if DUMP_DELAYED:  # the dump already holds the delayed detections (eval_vision stores feat after the delay queue)
                f = d["feat"][t]
            else:
                f = d["feat"][td] if td >= 0 else zero
            ekf.update(f, d["gi"][t], d["mc"][t], d["mn"][t], d["last"][t], delay=D)
        gi = d["gi"][t]
        true_c = d["mc"][t][:, 0] + offs[ar, gi.clamp(max=offs.shape[1] - 1)]
        est_c = d["mc"][t][:, 0] + ekf.dg[ar, gi.clamp(max=ekf.dg.shape[1] - 1)]
        perr.append((ekf.p - d["gtp"][t]).norm(dim=-1))
        gerr.append((est_c - true_c).norm(dim=-1))
        dist.append((true_c - d["gtp"][t]).norm(dim=-1))
        if d["done"][t].any() and t + 1 < T:
            reset(torch.nonzero(d["done"][t]).flatten(), t + 1)
    return torch.stack(perr), torch.stack(gerr), torch.stack(dist), (torch.stack(nmr) if nmr else None)


def summ(label, P, G, Dd, d, nmr):
    T, N = d["done"].shape
    es = torch.zeros_like(d["done"]); es[1:] = d["done"][:-1]; es[0] = True
    tt = torch.arange(T)[:, None].expand(T, N)
    st = torch.where(es, tt, torch.zeros_like(tt)).cummax(0).values
    m = d["valid"] & ((tt - st) * DT > 0.5)
    out = [label]
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 30)):
        s = m & (Dd >= lo) & (Dd < hi)
        if s.sum() > 20:
            out.append(f"{lo}-{hi}m pos p50/p90 {P[s].median():.2f}/{P[s].quantile(0.9):.2f} gate p50/p90 {G[s].median():.2f}/{G[s].quantile(0.9):.2f}")
    if nmr is not None:
        out.append(f"corner updates/step {nmr[m].mean():.2f}")
    print(" | ".join(out))


for f in files:
    d = torch.load(f)
    offs = torch.randn(d["done"].shape[1], 64, 3) * float(__import__("os").environ.get("OFFS", "0.25"))
    print(f)
    P, G, Dd, _ = replay("gn", d, offs)
    summ("  ekf5 +GateNet feat ", P, G, Dd, d, None)
    P, G, Dd, nmr = replay("mr", d, offs)
    summ("  ekf8 +synthetic MR ", P, G, Dd, d, nmr)
