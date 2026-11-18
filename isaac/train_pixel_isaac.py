"""Pixel RL in Isaac Lab: recurrent PPO from RGB + IMU, no teacher, asymmetric critic.

Run: scripts/run_isaaclab.sh isaac/train_pixel_isaac.py --task runs/t1 --n-envs 256 --iters 500 --out runs/p1
``--task`` only supplies the task/DR configuration (config.json of a teacher run); its weights are NOT used.
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--n-envs", type=int, default=256)
parser.add_argument("--iters", type=int, default=500)
parser.add_argument("--T", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--aug", type=float, default=1.0)
parser.add_argument("--aux", type=float, default=0.5, help="auxiliary supervised loss weight (0 = pure RL)")
parser.add_argument("--img-h", type=int, default=48)
parser.add_argument("--img-w", type=int, default=64)
parser.add_argument("--env-spacing", type=float, default=150.0)
parser.add_argument("--sensor-ramp", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="runs/p1")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/isaac")
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.pixel_ppo import run_pixel_ppo  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402
from vision_rig import VisionRig  # noqa: E402

a = args_cli
torch.manual_seed(a.seed)
meta = json.loads((Path("/workspace/DroneRacing") / a.task / "config.json").read_text())
cfg = make_cfg(dict(meta["args"]), a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
cfg.mid_init_prob = 0.5
target_scale = cfg.sensors.scale

rig = VisionRig(a.n_envs, a.img_h, a.img_w, cam_tilt=cfg.cam_tilt)
sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0", env_spacing=a.env_spacing, extra_cfg=rig.attach,
                         ground_size=a.env_spacing * (int(a.n_envs ** 0.5) + 2))
rig.bind(sim, scene)
env = RaceEnv(cfg, seed=a.seed, quad_factory=IsaacQuad.factory(sim, scene))
policy_dt = cfg.dt * cfg.decimation
rig.update(env)


def ramp(it, total):
    if a.sensor_ramp > 0:
        cfg.sensors.scale = target_scale * min(1.0, it / a.sensor_ramp)


run_pixel_ppo(env, lambda: rig.render(policy_dt), (a.img_h, a.img_w), Path("/workspace/DroneRacing") / a.out,
              a.iters, a.T, a.lr, a.aug, a.aux, post_step=rig.update, on_iter=ramp)
print("PIXEL PPO DONE", flush=True)
app.close()
