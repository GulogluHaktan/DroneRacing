"""Held-out evaluation: first episode of each env on fresh seeds (unseen tracks + vehicles).

Reports success, crash, lap time and mean speed overall and per thrust-to-weight bucket.
Run: PYTHONPATH=. ~/isaac-env/bin/python -I scripts/eval_state.py --run runs/a1_dr1_params
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from droneracing import EnvCfg, RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402


@torch.no_grad()
def evaluate(run: Path, n_envs=2048, seed=12345, dr_scale=None, max_iters=2000, overrides=None, cfg_hook=None):
    meta = json.loads((run / "config.json").read_text())
    args = dict(meta["args"])
    if dr_scale is not None:
        args["dr_scale"] = dr_scale
    args.update(overrides or {})
    cfg = make_cfg(args, n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
    cfg.mid_init_prob = 0.0  # evaluation = full laps from the start line
    if cfg_hook:
        cfg_hook(cfg)
    import os as _os
    cfg.device = _os.environ.get("DR_DEVICE", cfg.device)
    env = RaceEnv(cfg, seed=seed)  # seed differs from training -> unseen tracks/vehicles
    if args.get("ekf_loop"):  # actor obs from the real gate filter + analytic detector (camera-line surrogate, droneracing/ekf_loop.py)
        from droneracing.ekf_loop import EkfLoopEnv
        from droneracing.ekf_loop import DR_BAD_DAY, DR_TRAIN
        dr_ = {"train": DR_TRAIN, "bad_day": DR_BAD_DAY}.get(args.get("ekf_loop_dr") or "")
        env = EkfLoopEnv(env, recall=float(args.get("ekf_loop_recall") or 0.95), dr=dr_, calib=args.get("ekf_loop_calib") or "")
    pm = meta.get("ppo", {})
    ac = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg(hidden=tuple(pm.get("hidden", (256, 256, 128))), critic_hidden=pm.get("critic_hidden"))).to(cfg.device)
    ac.load_state_dict(torch.load(run / "model.pt", map_location=cfg.device)["ac"])
    ac.eval()
    obs, _ = env.compute_obs()
    seen = torch.zeros(n_envs, dtype=torch.bool, device=cfg.device)
    rec = {k: torch.zeros(n_envs, device=cfg.device) for k in ("fin", "crash", "len", "twr", "tlen", "gates", "ng", "vmax", "cause", "vcrash")}
    vmax = torch.zeros(n_envs, device=cfg.device)
    jit_sum = torch.zeros(n_envs, device=cfg.device)
    jit_n = torch.zeros(n_envs, device=cfg.device)
    prev = None
    for _ in range(max_iters):
        act = ac.actor(ac.pn(obs))
        if prev is not None:
            alive = ~seen
            jit_sum += alive * (act.clamp(-1, 1) - prev).abs().mean(-1)
            jit_n += alive.float()
        prev = act.clamp(-1, 1)
        obs, _, _, _, done, info = env.step(act)
        vmax = torch.maximum(vmax, env.quad.vel.norm(dim=-1) * (~seen).float())
        new = done & ~seen
        if new.any():
            rec["fin"][new] = info["finished"][new].float()
            rec["crash"][new] = info["crashed"][new].float()
            rec["len"][new] = info["ep_len"][new].float() * cfg.dt * cfg.decimation
            rec["twr"][new] = info["twr"][new]
            rec["tlen"][new] = info["track_len"][new]
            rec["gates"][new] = info["gates_passed"][new].float()
            rec["ng"][new] = info["n_gates"][new].float()
            rec["vmax"][new] = vmax[new]
            rec["cause"][new] = info["cause"][new].float()
            rec["vcrash"][new] = info["speed"][new]
            seen |= new
        if seen.all():
            break
    rec["jitter"] = jit_sum / jit_n.clamp_min(1)
    return {k: v.cpu() for k, v in rec.items()}


def summarize(r, label="all", mask=None):
    m = torch.ones_like(r["fin"], dtype=torch.bool) if mask is None else mask
    if m.sum() == 0:
        return
    fin = m & (r["fin"] > 0)
    lap = r["len"][fin]
    speed = (r["tlen"][fin] / r["len"][fin].clamp_min(1e-3))
    print(
        f"{label:>14} | n {int(m.sum()):4d} | success {r['fin'][m].mean()*100:5.1f}% | crash {r['crash'][m].mean()*100:5.1f}% | "
        f"gates {(r['gates'][m]/r['ng'][m]).mean()*100:5.1f}% | lap {lap.mean() if len(lap) else float('nan'):5.1f}s | "
        f"path-speed {speed.mean() if len(speed) else float('nan'):4.1f} m/s (p90 {speed.quantile(0.9) if len(speed) else float('nan'):4.1f}) | "
        + (f"mean-speed {r['mspeed'][fin].mean():4.1f} m/s | " if "mspeed" in r and len(lap) else "")
        + (f"peak {r['vmax'][m].mean():4.1f} (p90 {r['vmax'][m].quantile(0.9):4.1f}, max {r['vmax'][m].max():4.1f}) | " if "vmax" in r else "")
        + f"action jitter {r['jitter'][m].mean():.3f}/step"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-envs", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--profile", default=None, help="override the track/vehicle profile (e.g. a2rl_hall)")
    ap.add_argument("--hall-prob", type=float, default=None)
    ap.add_argument("--drag-prob", type=float, default=None)
    ap.add_argument("--gate-box", action="store_true", default=None, help="evaluate with the oriented body box at the gate")
    ap.add_argument("--gate-body", action="store_true", default=None, help="evaluate with the drone body size at the gate")
    ap.add_argument("--cam-hz", type=float, default=None, help="camera frame rate (sightings only every policy_hz/cam_hz steps)")
    ap.add_argument("--corner-vis", action="store_true", default=None, help="sightings need gate corners in the image (partial-gate model)")
    ap.add_argument("--max-iters", type=int, default=4500, help="policy steps; must cover max_episode_s (40 s = 4000 steps at 100 Hz), else long episodes are dropped")
    ap.add_argument("--sensor-scale", type=float, default=None, help="override the sensing-noise scale (0 = perfect state and gates)")
    ap.add_argument("--tail-prob", type=float, default=None, help="override the heavy-tail gate-error probability per gate")
    ap.add_argument("--tail-mult", type=float, default=None)
    ap.add_argument("--resid-bank", default=None, help="evaluate under an empirical residual bank instead of the parametric errors")
    ap.add_argument("--fixed-prob", type=float, default=None, help="fraction of A2RL 2025 track replicas (1 = only that track)")
    ap.add_argument("--fixed-jitter", type=float, nargs=2, default=None)
    ap.add_argument("--fixed-scale", type=float, nargs=2, default=None)
    ap.add_argument("--fixed-mirror", type=float, default=None)
    ap.add_argument("--split-noise", type=float, nargs="+", default=None)
    ap.add_argument("--resid-prob", type=float, default=None)
    ap.add_argument("--ekf-loop", action="store_true", default=None, help="camera-line surrogate: actor observes GateEKF5 fed by simulated detections")
    ap.add_argument("--ekf-loop-recall", type=float, default=None)
    ap.add_argument("--ekf-loop-calib", default=None, help="detector table measured in the Isaac loop")
    ap.add_argument("--ekf-loop-dr", default=None, choices=["train", "bad_day"], help="perception randomization in the surrogate")
    ap.add_argument("--resid-scale", type=float, default=None, help="multiplier of the bank residuals (perception error scale)")
    ap.add_argument("--opp-prob", type=float, default=None, help="share of the episodes with scripted opponents (policy must have been trained with --opp-max)")
    ap.add_argument("--opp-speed", type=float, nargs=2, default=None)
    a = ap.parse_args()
    ov = {k: v for k, v in (("profile", a.profile), ("hall_prob", a.hall_prob), ("drag_prob", a.drag_prob), ("cam_hz", a.cam_hz), ("gate_body", a.gate_body), ("gate_box", a.gate_box), ("corner_vis", a.corner_vis), ("sensor_scale", a.sensor_scale), ("tail_prob", a.tail_prob), ("tail_mult", a.tail_mult), ("resid_bank", a.resid_bank),
                                ("fixed_prob", a.fixed_prob), ("fixed_jitter", a.fixed_jitter), ("fixed_scale", a.fixed_scale), ("fixed_mirror", a.fixed_mirror), ("split_noise", a.split_noise), ("resid_prob", a.resid_prob), ("resid_scale", a.resid_scale), ("ekf_loop", a.ekf_loop), ("ekf_loop_recall", a.ekf_loop_recall), ("ekf_loop_dr", a.ekf_loop_dr), ("ekf_loop_calib", a.ekf_loop_calib), ("opp_prob", a.opp_prob), ("opp_speed", a.opp_speed)) if v is not None}
    r = evaluate(Path(a.run), a.n_envs, a.seed, overrides=ov, max_iters=a.max_iters)
    summarize(r, "ALL")
    cr = r["crash"] > 0
    if cr.any():  # what kills the policy: share of crashed drones per cause and their speed at the crash
        names = {1: "frame", 2: "pole", 4: "ground", 5: "out", 6: "opponent", 0: "decoy/other"}
        print("  crash causes: " + ", ".join(f"{nm} {((r['cause'] == c) & cr).float().sum() / cr.sum() * 100:4.1f}% (v {r['vcrash'][(r['cause'] == c) & cr].mean():4.1f})"
                                            for c, nm in names.items() if ((r['cause'] == c) & cr).any()))
    for lo, hi in ((1.8, 3), (3, 4.5), (4.5, 6), (6, 8.1)):
        summarize(r, f"TWR {lo}-{hi}", (r["twr"] >= lo) & (r["twr"] < hi))
