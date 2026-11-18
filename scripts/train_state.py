"""Phase A training: state-based PPO with vehicle + track domain randomization.

Run: PYTHONPATH=. ~/isaac-env/bin/python -I scripts/train_state.py --iters 500 --out runs/a0
"""
import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.ppo import PPO, PPOCfg  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=4096)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--dr-scale", type=float, default=1.0, help="vehicle randomization range scale in [0,1]")
    p.add_argument("--hide-params", action="store_true", help="actor does not see vehicle parameters")
    p.add_argument("--vis-obs", action="store_true", help="gates known only after the camera saw them (FOV test), held with odometry drift")
    p.add_argument("--map-obs", action="store_true", help="approximate track map known from the start (EKF-style localisation), sightings refine the gates")
    p.add_argument("--no-obstacles", action="store_true", help="no poles (A2RL tracks are gates only)")
    p.add_argument("--sensor-profile", default=None, choices=("vision", "closedloop", "ekf"), help="estimator-grade sensing errors")
    p.add_argument("--obs-history", type=int, default=0, help="append K past (vel, gyro, action) frames to the actor obs")
    p.add_argument("--profile", choices=("wide", "racing", "a2rl", "a2rl_long", "a2rl_hall", "own_drone"), default="a2rl")
    p.add_argument("--sensor-scale", type=float, default=None, help="sensing/actuation imperfection severity (default 1 for a2rl, 0 otherwise)")
    p.add_argument("--v-max", type=float, default=None, help="cap on progress speed [m/s] (reward), 0/None = off")
    p.add_argument("--v-max-start", type=float, default=None, help="curriculum: start v_max, linearly raised to --v-max")
    p.add_argument("--sensor-ramp", type=int, default=0, help="curriculum: ramp sensor scale 0 -> target over N iterations")
    p.add_argument("--markers", action="store_true", help="privileged marker tokens (teacher)")
    p.add_argument("--landing", action="store_true", help="must land after the last gate")
    p.add_argument("--fork-prob", type=float, default=0.0)
    p.add_argument("--split-prob", type=float, default=0.0, help="split-S module probability per gate")
    p.add_argument("--loop-prob", type=float, default=0.0, help="stacked loop module probability per gate")
    p.add_argument("--yaw-cmd", type=float, default=0.0, help="penalty on the squared yaw-rate command")
    p.add_argument("--yaw-diff", type=float, default=0.0, help="penalty on the step-to-step change of the yaw-rate command")
    p.add_argument("--n-obs-gates", type=int, default=None, help="number of upcoming gates in the observation (default 2)")
    p.add_argument("--init-ng", type=int, default=2, help="gates in the --init policy (new gate columns are zero-initialised)")
    p.add_argument("--bias-rand", type=float, nargs=2, default=None, help="LO HI: log-uniform per-episode multiplier of map/velocity biases")
    p.add_argument("--soft-frac", type=float, default=0.0, help="start fraction of episodes with soft gate-frame collisions (penalty, no termination)")
    p.add_argument("--soft-ramp", type=int, default=0, help="iterations over which --soft-frac is linearly annealed to 0 (hard collisions)")
    p.add_argument("--policy-hz", type=float, default=None, help="policy rate [Hz] (default 50); per-step rewards / noise are rescaled, gamma/lambda/n_steps follow --gamma/--n-steps at 50 Hz")
    p.add_argument("--motor-direct", action="store_true", help="action = 4 motor commands (no rate loop / mixer), MonoRace G&CNet style")
    p.add_argument("--hall-prob", type=float, default=None, help="fraction of hall-bounded closed-lap tracks (profile a2rl_hall default 0.5)")
    p.add_argument("--hidden", type=str, default="256,256,128", help="actor MLP widths")
    p.add_argument("--critic-hidden", type=str, default=None, help="critic MLP widths (default = --hidden)")
    p.add_argument("--ekf-loop", action="store_true", help="estimator-in-the-loop training: actor obs = GateEKF5 output from simulated detections (own camera, 30 Hz, 50 ms)")
    p.add_argument("--ekf-loop-recall", type=float, default=0.95)
    p.add_argument("--ekf-loop-calib", default="", help="detector table measured in the Isaac loop (scripts/fit_detector.py)")
    p.add_argument("--ekf-loop-dr", action="store_true", help="perception domain randomization in the loop (detector quality, delay mismatch, mount residual, lost frames)")
    p.add_argument("--corner-vis-p", type=float, nargs=3, default=None, help="detection probability with 4 / 3 / 2 corners in the image")
    p.add_argument("--speed-pen", type=float, default=None, help="per-second penalty on the speed above --speed-cap (safe model)")
    p.add_argument("--speed-cap", type=float, default=None)
    p.add_argument("--opp-max", type=int, default=0, help="scripted opponent drones per episode (1..N, no camera/RL); 0 = single-drone racing")
    p.add_argument("--opp-prob", type=float, default=None, help="share of the episodes that have opponents (default 1.0)")
    p.add_argument("--opp-speed", type=float, nargs=2, default=None, help="opponent cruise speed range [m/s] (default 6 13)")
    p.add_argument("--opp-near", type=float, default=None, help="per-step proximity penalty weight for an opponent within 1 m (surface gap)")
    p.add_argument("--slow-pen", type=float, default=None, help="per-second penalty on the speed BELOW --slow-min (fast model), not applied in the first --slow-grace seconds")
    p.add_argument("--slow-min", type=float, default=None)
    p.add_argument("--mid-prob", type=float, default=None, help="fraction of resets that start in front of a random gate (default 0.5 for the a2rl-like profiles)")
    p.add_argument("--mid-speed", type=float, nargs=2, default=None, help="LO HI speed [m/s] of the mid-track starts (default 0 10)")
    p.add_argument("--mid-focus", type=int, nargs="+", default=None, help="gate indices that the mid-track starts prefer")
    p.add_argument("--mid-focus-prob", type=float, default=None, help="share of the mid-track starts that use a focus gate (default 0.5)")
    p.add_argument("--smooth-mult", type=float, default=None, help="multiplier of the body-rate and action-change penalties (smoother flying)")
    p.add_argument("--slow-grace", type=float, default=None)
    p.add_argument("--align-pen", type=float, default=None, help="penalty on the gate-entry angle above 10 deg (per pass, scaled by angle/45 deg)")
    p.add_argument("--gate-box", action="store_true", help="geometry-aware gate passage: oriented body box (attitude at the crossing) instead of the sphere; use with --gate-body")
    p.add_argument("--gate-body", action="store_true", help="gate passage checks the drone body (half opening - radius) instead of a point")
    p.add_argument("--perc-pen", type=float, default=0.0, help="penalty per radian per second when the next gate is >60 deg off the camera axis")
    p.add_argument("--gate-shrink", type=float, default=0.0, help="training only: shrink the half gate opening by this many metres (MonoRace-style tight gates)")
    p.add_argument("--corner-vis", action="store_true", help="sightings need the gate corners in the image (partial gates rarely detected)")
    p.add_argument("--tail-prob", type=float, default=0.0, help="probability that a gate has a heavy-tail perception error in an episode")
    p.add_argument("--tail-mult", type=float, default=1.0, help="error scale of such a gate")
    p.add_argument("--resid-bank", default="", help="Swift-style empirical residual bank (scripts/build_resid_bank.py); replaces the parametric state/gate errors")
    p.add_argument("--resid-scale", type=float, default=1.0)
    p.add_argument("--resid-prob", type=float, default=1.0, help="fraction of envs on the bank (rest parametric)")
    p.add_argument("--init-std", type=float, default=None, help="policy std after --init (default: PPOCfg.init_log_std); keep low for a polishing run")
    p.add_argument("--fixed-prob", type=float, default=0.0, help="fraction of episodes on the A2RL 2025 track replica (track.a2rl_2025_lap)")
    p.add_argument("--fixed-jitter", type=float, nargs=2, default=None, help="POS_M YAW_RAD per-gate jitter of the replica")
    p.add_argument("--fixed-scale", type=float, nargs=2, default=None, help="LO HI whole-layout scale")
    p.add_argument("--fixed-mirror", type=float, default=0.0, help="probability of the mirrored layout")
    p.add_argument("--split-noise", type=float, nargs="+", default=None, help="sighting-noise multipliers at the split-S lower gate and the gates after it")
    p.add_argument("--age-obs", action="store_true", help="append time-since-last-sighting of the observed gates to the policy obs")
    p.add_argument("--obs-delay", type=int, nargs=2, default=None, help="LO HI perception delay in policy steps (profile own_drone default 2 7)")
    p.add_argument("--cam-hz", type=float, default=None, help="camera frame rate; sightings only every policy_hz/cam_hz policy steps")
    p.add_argument("--drag-prob", type=float, default=None, help="fraction of straight drag-race corridors among random-walk tracks")
    p.add_argument("--gate-speed", type=float, default=0.0, help="reward on gate pass proportional to speed at the gate (speed/10, capped at 25 m/s)")
    p.add_argument("--finish-time", type=float, default=0.0)
    p.add_argument("--crash-pen", type=float, default=None, help="crash penalty (RewardCfg.crash, default 10)")
    p.add_argument("--center-rew", type=float, default=None, help="gate-centre bonus on pass (RewardCfg.center; a2rl profile sets 1)")
    p.add_argument("--ent-coef", type=float, default=0.0, help="entropy bonus (keeps exploration alive)")
    p.add_argument("--init", default=None, help="warm start from a run dir with the same observation layout (std is reset)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="runs/a0")
    p.add_argument("--log-every", type=int, default=10)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    cfg = make_cfg(vars(a), a.n_envs)
    cfg.soft_frac = a.soft_frac  # training only (evaluation always uses hard collisions)
    cfg.gate_shrink = a.gate_shrink  # training only (evaluation uses the physical opening)
    import os as _os
    cfg.device = _os.environ.get("DR_DEVICE", cfg.device)  # CPU smoke tests while the GPU renders
    env = RaceEnv(cfg, seed=a.seed)
    if a.ekf_loop:  # actor observes the real gate filter fed by an analytic camera + detector model (droneracing/ekf_loop.py)
        from droneracing.ekf_loop import EkfLoopEnv
        from droneracing.ekf_loop import DR_TRAIN
        env = EkfLoopEnv(env, recall=a.ekf_loop_recall, dr=DR_TRAIN if a.ekf_loop_dr else None, calib=a.ekf_loop_calib)
    ratio = cfg.dt * cfg.decimation / 0.02  # policy step relative to the 50 Hz default
    ppo = PPO(env, PPOCfg(gamma=a.gamma**ratio, lam=0.95**ratio, n_steps=max(32, round(32 / ratio)), ent_coef=a.ent_coef,
                       hidden=tuple(int(x) for x in a.hidden.split(",")), critic_hidden=tuple(int(x) for x in a.critic_hidden.split(",")) if a.critic_hidden else None))
    if a.init:
        sd = torch.load(Path(a.init) / "model.pt", map_location=cfg.device)["ac"]
        if cfg.n_obs_gates != a.init_ng:  # insert zero columns for the extra gate blocks (policy starts identical to the init one)
            gd, pos = env.gate_dim, 13 + a.init_ng * env.gate_dim
            add = (cfg.n_obs_gates - a.init_ng) * gd
            for k in ("actor.0.weight", "critic.0.weight"):
                w = sd[k]
                sd[k] = torch.cat((w[:, :pos], torch.zeros(w.shape[0], add, device=w.device, dtype=w.dtype), w[:, pos:]), 1)
            for k, fill in (("pn.mean", 0.0), ("pn.var", 1.0), ("cn.mean", 0.0), ("cn.var", 1.0)):
                w = sd[k]
                sd[k] = torch.cat((w[:pos], torch.full((add,), fill, device=w.device, dtype=w.dtype), w[pos:]), 0)
        if cfg.age_obs and sd["actor.0.weight"].shape[1] == env.policy_dim - cfg.n_obs_gates:  # zero columns for the appended age inputs
            add = cfg.n_obs_gates
            w = sd["actor.0.weight"]
            sd["actor.0.weight"] = torch.cat((w, torch.zeros(w.shape[0], add, device=w.device, dtype=w.dtype)), 1)
            for k, fill in (("pn.mean", 0.0), ("pn.var", 1.0)):
                sd[k] = torch.cat((sd[k], torch.full((add,), fill, device=sd[k].device, dtype=sd[k].dtype)), 0)
        if cfg.opp_max > 0 and sd["actor.0.weight"].shape[1] < env.policy_dim:  # zero columns for the appended opponent block (policy starts identical)
            for wk, nk, tot in (("actor.0.weight", "pn", env.policy_dim), ("critic.0.weight", "cn", env.critic_dim)):
                add = tot - sd[wk].shape[1]
                w = sd[wk]
                sd[wk] = torch.cat((w, torch.zeros(w.shape[0], add, device=w.device, dtype=w.dtype)), 1)
                for k, fill in ((nk + ".mean", 0.0), (nk + ".var", 1.0)):
                    sd[k] = torch.cat((sd[k], torch.full((add,), fill, device=sd[k].device, dtype=sd[k].dtype)), 0)
        ppo.ac.load_state_dict(sd)
        with torch.no_grad():
            ppo.ac.log_std.fill_(ppo.cfg.init_log_std if a.init_std is None else __import__("math").log(a.init_std))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"env": asdict(cfg), "ppo": asdict(ppo.cfg), "args": vars(a)}, indent=1))
    f = open(out / "log.csv", "w", newline="")
    writer = None

    def log(row):
        nonlocal writer
        if writer is None:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        print(
            f"it {row['it']:4d} | {row['sps']/1e3:6.0f}k sps | ret {row['ret']:7.1f} | gates {row['gates']:4.1f} "
            f"({row['frac']*100:3.0f}%) | finish {row['finish']*100:3.0f}% | crash {row['crash']*100:3.0f}% | "
            f"v {row['speed']:4.1f} | std {row['std']:.2f} | kl {row['kl']:.4f}",
            flush=True,
        )

    def ckpt(it, algo):
        torch.save({"ac": algo.ac.state_dict(), "it": it}, out / "model.pt")

    target_scale = cfg.sensors.scale
    v_end = cfg.reward.v_max

    def curriculum(it, total, algo):
        if a.soft_ramp > 0:
            cfg.soft_frac = a.soft_frac * max(0.0, 1.0 - it / a.soft_ramp)
        if a.sensor_ramp > 0:
            cfg.sensors.scale = target_scale * min(1.0, it / a.sensor_ramp)
        if a.v_max_start is not None and v_end > 0:
            cfg.reward.v_max = a.v_max_start + (v_end - a.v_max_start) * min(1.0, it / max(1, total * 0.6))

    ppo.train(a.iters, log_every=a.log_every, ckpt=ckpt, log_fn=log, on_iter=curriculum)


if __name__ == "__main__":
    main()
