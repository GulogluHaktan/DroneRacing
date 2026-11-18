"""DAgger distillation of the privileged teacher into the RGB + IMU student, entirely in Isaac Lab
(PhysX vehicle physics + RTX TiledCamera). No analytic camera is involved.

Run: scripts/run_isaaclab.sh isaac/train_student_isaac.py --teacher runs/t1 --n-envs 256 --iters 200 --out runs/s1 --enable_cameras
The teacher run must have been trained with --markers (its config.json defines the task).
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--teacher", type=str, required=True)
parser.add_argument("--n-envs", type=int, default=256)
parser.add_argument("--iters", type=int, default=200)
parser.add_argument("--T", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--beta0", type=float, default=1.0)
parser.add_argument("--beta-iters", type=int, default=60)
parser.add_argument("--aug", type=float, default=1.0)
parser.add_argument("--img-h", type=int, default=48)
parser.add_argument("--img-w", type=int, default=64)
parser.add_argument("--env-spacing", type=float, default=150.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="runs/s1")
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
from droneracing.dagger import run_dagger  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402
from droneracing.student import StudentNet  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402
from vision_rig import VisionRig  # noqa: E402

a = args_cli
torch.manual_seed(a.seed)
teacher_dir = Path("/workspace/DroneRacing") / a.teacher
meta = json.loads((teacher_dir / "config.json").read_text())
cfg = make_cfg(dict(meta["args"]), a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
cfg.mid_init_prob = 0.5

rig = VisionRig(a.n_envs, a.img_h, a.img_w, cam_tilt=cfg.cam_tilt)
sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0", env_spacing=a.env_spacing, extra_cfg=rig.attach,
                        ground_size=150.0 * (int(a.n_envs ** 0.5) + 2))
rig.bind(sim, scene)
env = RaceEnv(cfg, seed=a.seed, quad_factory=IsaacQuad.factory(sim, scene))
dev = env.device

teacher = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg()).to(dev)
teacher.load_state_dict(torch.load(teacher_dir / "model.pt", map_location=dev)["ac"])
teacher.eval()
student = StudentNet(img_hw=(a.img_h, a.img_w)).to(dev)

policy_dt = cfg.dt * cfg.decimation
rig.update(env)


def render():
    return rig.render(policy_dt)


def after_step(e):
    rig.update(e)  # gates / markers follow the (possibly reset) track before the next frame


run_dagger(env, render, teacher, student, Path("/workspace/DroneRacing") / a.out, a.iters, a.T, a.lr,
           a.beta0, a.beta_iters, a.aug, post_step=after_step)
print("DAGGER DONE", flush=True)
app.close()
