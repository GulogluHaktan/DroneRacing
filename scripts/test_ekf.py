"""Map-based EKF in the torch env with the analytic detector model: error vs time, and closed-loop success when the policy flies on it.

Run: ~/isaac-env/bin/python -I scripts/test_ekf.py --run runs/t8snap [--closed] [--det-noise 1.0] [--map-err 0.25]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.detector_model import sim_detections  # noqa: E402
from droneracing.ekf import GateEKF  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402
from droneracing.quad import quat_exp, quat_mul, quat_to_rot  # noqa: E402
from eval_state import summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--n-envs", type=int, default=512)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--closed", action="store_true", help="policy flies on EKF outputs (else on the env's oracle obs)")
ap.add_argument("--det-noise", type=float, default=1.0)
ap.add_argument("--map-err", type=float, default=0.25, help="map position error std [m] per gate")
ap.add_argument("--map-yaw", type=float, default=0.09, help="map yaw error std [rad] per gate")
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--ekf-args", type=str, default="{}", help="json overrides for GateEKF")
a = ap.parse_args()
run = Path(a.run)
meta = json.loads((run / "config.json").read_text())
cfg = make_cfg(dict(meta["args"]), a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
cfg.mid_init_prob = 0.0
cfg.sensors.gate_bias, cfg.sensors.gate_yaw_bias = a.map_err, a.map_yaw
env = RaceEnv(cfg, seed=a.seed)
dev = env.device
n, K = a.n_envs, cfg.obs_history
H, W = 192, 256
dt = cfg.dt * cfg.decimation
ac = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg()).to(dev)
ac.load_state_dict(torch.load(run / "model.pt", map_location=dev)["ac"])
ac.eval()
ekf = GateEKF(n, dev, sig_map=max(a.map_err, 0.05), **json.loads(a.ekf_args))
g = torch.Generator(device=dev).manual_seed(a.seed)


def ekf_reset(ids):
    k = len(ids)
    q = quat_mul(env.quad.quat[ids], quat_exp(torch.randn(k, 3, device=dev) * 0.02))
    ekf.reset(ids, env.quad.pos[ids] + torch.randn(k, 3, device=dev) * 0.05, env.quad.vel[ids], q)


def map_gates(J=3):
    last = env._last_slot()
    idx = [torch.minimum(env.gate_idx + j, last) for j in range(J)]
    cs, ns = zip(*[env._gate_frame(i, noisy=True)[:2] for i in idx])
    return torch.stack(cs, 1), torch.stack(ns, 1), last


ekf_reset(torch.arange(n, device=dev))
obs, _ = env.compute_obs()
hist = torch.zeros(n, K, 10, device=dev)
age = torch.zeros(n, device=dev)
seen = torch.zeros(n, dtype=torch.bool, device=dev)
rec = {k: torch.zeros(n, device=dev) for k in ("fin", "crash", "len", "twr", "tlen", "gates", "ng", "jitter")}
bins = [(0, 2), (2, 5), (5, 8), (8, 12), (12, 20), (20, 60)]
acc = {b: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for b in bins}
dbins = [(0, 3), (3, 6), (6, 10), (10, 20)]
dacc = {b: [0.0, 0.0] for b in dbins}  # next-gate rel error by true distance  # n, vel err, att err deg, gate0 rel err (true gate), pos err, accepted
prev = None
with torch.no_grad():
    for step in range(a.steps):
        imu = env.imu_obs()
        ekf.predict(imu, dt)
        feat = sim_detections(env, g, H, W, noise_scale=a.det_noise)
        mc, mn, last = map_gates()
        ekf.update(feat, env.gate_idx, mc, mn, last)
        vel_b, up, rel, nrm = ekf.outputs(env.gate_idx, last, mc, mn)
        # errors against the truth
        R = quat_to_rot(env.quad.quat)
        Rt = R.transpose(1, 2)
        vtrue = (Rt @ env.quad.vel[..., None])[..., 0]
        c_true, _, _ = env._gate_frame(torch.minimum(env.gate_idx, last))
        rtrue = (Rt @ (c_true - env.quad.pos)[..., None])[..., 0]
        ev = (vel_b - vtrue).norm(dim=-1)
        ea = torch.acos((torch.nn.functional.normalize(up, dim=-1) * R[:, 2, :]).sum(-1).clamp(-1, 1)) * 57.3
        eg = (rel[:, 0] - rtrue).norm(dim=-1)
        ep = (ekf.p - env.quad.pos).norm(dim=-1)
        alive = ~seen
        for (lo, hi) in bins:
            m = alive & (age * dt >= lo) & (age * dt < hi)
            if m.any():
                s = acc[(lo, hi)]
                s[0] += float(m.sum()); s[1] += float(ev[m].sum()); s[2] += float(ea[m].sum()); s[3] += float(eg[m].sum()); s[4] += float(ep[m].sum()); s[5] += float(ekf.accepted[m].sum())
        dtrue = rtrue.norm(dim=-1)
        for (lo, hi) in dbins:
            m = alive & (dtrue >= lo) & (dtrue < hi)
            if m.any():
                dacc[(lo, hi)][0] += float(m.sum()); dacc[(lo, hi)][1] += float(eg[m].sum())
        if a.closed:
            flag = torch.ones(n, 1, device=dev)
            parts = [vel_b / 10.0, up, imu[:, :3], env.prev_action]
            for j in range(2):
                parts += [rel[:, j] / 5.0, nrm[:, j], env.gate_size[:, None], flag]
            pol_obs = torch.cat(parts + [hist.reshape(n, -1)], -1)
            hist = torch.roll(hist, -1, dims=1)
            hist[:, -1] = torch.cat((vel_b / 10.0, imu[:, :3], env.prev_action), -1)
        else:
            pol_obs = obs
        act = ac.actor(ac.pn(pol_obs)).clamp(-1, 1)
        if prev is not None:
            rec["jitter"] += alive * (act - prev).abs().mean(-1)
        prev = act
        obs, _, _, _, done, info = env.step(act)
        age = torch.where(done, torch.zeros_like(age), age + 1)
        if done.any():
            ids = torch.nonzero(done).flatten()
            ekf_reset(ids)
            hist[ids] = 0.0
        new = done & ~seen
        if new.any():
            rec["fin"][new] = info["finished"][new].float()
            rec["crash"][new] = info["crashed"][new].float()
            rec["len"][new] = info["ep_len"][new].float() * dt
            rec["twr"][new] = info["twr"][new]
            rec["tlen"][new] = info["track_len"][new]
            rec["gates"][new] = info["gates_passed"][new].float()
            rec["ng"][new] = info["n_gates"][new].float()
            seen |= new
        if seen.all():
            break
rec["jitter"] = rec["jitter"] / max(1, step)
print(f"EKF {'closed-loop (policy on EKF)' if a.closed else 'open-loop (policy on oracle obs)'} | detector noise x{a.det_noise} | map err {a.map_err} m / {a.map_yaw} rad")
print("age s | n | vel err m/s | attitude err deg | next-gate rel err m | pos err m | accepted det/step")
for b, s in acc.items():
    if s[0] > 200:
        print(f"{b[0]:2d}-{b[1]:2d} | {int(s[0]):6d} | {s[1]/s[0]:.2f} | {s[2]/s[0]:.1f} | {s[3]/s[0]:.2f} | {s[4]/s[0]:.2f} | {s[5]/s[0]:.2f}")
print("next-gate rel err by distance: " + " | ".join(f"{b[0]}-{b[1]} m: {s[1]/max(s[0],1):.2f}" for b, s in dacc.items()))
summarize({k: v.cpu() for k, v in rec.items()}, "EKF-" + ("CL" if a.closed else "OL"))
