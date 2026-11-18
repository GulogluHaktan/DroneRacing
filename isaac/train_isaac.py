"""PPO training with Isaac Lab (PhysX) vehicle physics.

Run: scripts/run_isaaclab.sh isaac/train_isaac.py --n-envs 2048 --iters 300 --out runs/c1_isaac
"""
import argparse
import csv
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--n-envs", type=int, default=2048)
parser.add_argument("--iters", type=int, default=300)
parser.add_argument("--profile", choices=("wide", "racing", "a2rl"), default="a2rl")
parser.add_argument("--dr-scale", type=float, default=1.0)
parser.add_argument("--hide-params", action="store_true")
parser.add_argument("--sensor-scale", type=float, default=None)
parser.add_argument("--markers", action="store_true")
parser.add_argument("--landing", action="store_true")
parser.add_argument("--fork-prob", type=float, default=0.0)
parser.add_argument("--split-prob", type=float, default=0.0)
parser.add_argument("--loop-prob", type=float, default=0.0)
parser.add_argument("--finish-time", type=float, default=0.0)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n-steps", type=int, default=24)
parser.add_argument("--out", type=str, default="runs/c1_isaac")
parser.add_argument("--log-every", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app = AppLauncher(args_cli).app

from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/isaac")
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.ppo import PPO, PPOCfg  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402

a = args_cli
torch.manual_seed(a.seed)
cfg = make_cfg(vars(a), a.n_envs)
sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0")
env = RaceEnv(cfg, seed=a.seed, quad_factory=IsaacQuad.factory(sim, scene))
ppo = PPO(env, PPOCfg(gamma=a.gamma, n_steps=a.n_steps))

out = Path("/workspace/DroneRacing") / a.out
out.mkdir(parents=True, exist_ok=True)
args_dict = {k: v for k, v in vars(a).items() if isinstance(v, (int, float, str, bool))}
(out / "config.json").write_text(json.dumps({"env": asdict(cfg), "ppo": asdict(ppo.cfg), "args": args_dict}, indent=1))
f = open(out / "log.csv", "w", newline="")
writer = None


def log(row):
    global writer
    if writer is None:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
    writer.writerow(row)
    f.flush()
    print(
        f"it {row['it']:4d} | {row['sps']/1e3:6.1f}k sps | ret {row['ret']:7.1f} | gates {row['gates']:4.1f} "
        f"({row['frac']*100:3.0f}%) | finish {row['finish']*100:3.0f}% | crash {row['crash']*100:3.0f}% | "
        f"v {row['speed']:4.1f} | std {row['std']:.2f}",
        flush=True,
    )


def ckpt(it, algo):
    torch.save({"ac": algo.ac.state_dict(), "it": it}, out / "model.pt")


ppo.train(a.iters, log_every=a.log_every, ckpt=ckpt, log_fn=log)
app.close()
