"""Evaluate a policy trained in the torch env inside Isaac Lab physics (sim-to-sim check).

Run: scripts/run_isaaclab.sh isaac/eval_isaac.py --run runs/a1_dr1_params --n-envs 512
"""
import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--n-envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=12345)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app = AppLauncher(args_cli).app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/scripts")
from droneracing import EnvCfg, RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402
from eval_state import summarize  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing/isaac")
from isaac_quad import IsaacQuad, build_scene  # noqa: E402

run = Path("/workspace/DroneRacing") / args_cli.run
meta = json.loads((run / "config.json").read_text())
cfg = make_cfg(dict(meta["args"]), args_cli.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
cfg.mid_init_prob = 0.0  # evaluation = full laps from the start line

sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0")
env = RaceEnv(cfg, seed=args_cli.seed, quad_factory=IsaacQuad.factory(sim, scene))
ac = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg()).to(cfg.device)
ac.load_state_dict(torch.load(run / "model.pt", map_location=cfg.device)["ac"])
ac.eval()

n = cfg.n_envs
obs, _ = env.compute_obs()
seen = torch.zeros(n, dtype=torch.bool, device=cfg.device)
rec = {k: torch.zeros(n, device=cfg.device) for k in ("fin", "crash", "len", "twr", "tlen", "gates", "ng")}
jit_sum, jit_n, prev = torch.zeros(n, device=cfg.device), torch.zeros(n, device=cfg.device), None
t0, steps = time.time(), 0
with torch.no_grad():
    for _ in range(1500):
        act = ac.actor(ac.pn(obs)).clamp(-1, 1)
        if prev is not None:
            jit_sum += (~seen) * (act - prev).abs().mean(-1)
            jit_n += (~seen).float()
        prev = act
        obs, _, _, _, done, info = env.step(act)
        steps += 1
        new = done & ~seen
        if new.any():
            rec["fin"][new] = info["finished"][new].float()
            rec["crash"][new] = info["crashed"][new].float()
            rec["len"][new] = info["ep_len"][new].float() * cfg.dt * cfg.decimation
            rec["twr"][new] = info["twr"][new]
            rec["tlen"][new] = info["track_len"][new]
            rec["gates"][new] = info["gates_passed"][new].float()
            rec["ng"][new] = info["n_gates"][new].float()
            seen |= new
        if seen.all():
            break
dt_wall = time.time() - t0
print(f"policy steps {steps}, wall {dt_wall:.1f}s -> {steps*n/dt_wall:.0f} env-steps/s (Isaac physics, {n} envs)", flush=True)
rec["jitter"] = jit_sum / jit_n.clamp_min(1)
r = {k: v.cpu() for k, v in rec.items()}
summarize(r, "ISAAC ALL")
for lo, hi in ((1.8, 3), (3, 4.5), (4.5, 6), (6, 8.1)):
    summarize(r, f"TWR {lo}-{hi}", (r["twr"] >= lo) & (r["twr"] < hi))
app.close()
