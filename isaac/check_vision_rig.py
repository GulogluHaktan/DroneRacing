"""First thing to run when the GPU is free: does the Isaac vision rig show what RaceEnv believes?

Checks (prints PASS/FAIL, saves montage runs/vision_check.png):
 1. facing gate 0 from 3 m, the orange frame fills a visible fraction of the image;
 2. marker panel colour matches the token (turn colour);
 3. envs 150 m apart do not see each other (cameras at different origins give different images);
 4. gates move after reset (image changes when a new track is generated).
Run: scripts/run_isaaclab.sh isaac/check_vision_rig.py --enable_cameras
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--n-envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/isaac")
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.quad import yaw_to_quat  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402
from vision_rig import TURN_RGB, VisionRig  # noqa: E402

cfg = make_cfg({"profile": "a2rl", "sensor_scale": 0.0, "markers": True, "fork_prob": 0.5}, args_cli.n_envs)
cfg.mid_init_prob = 0.0
rig = VisionRig(args_cli.n_envs, 96, 128, cam_tilt=0.0)
sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0", env_spacing=150.0, extra_cfg=rig.attach)
rig.bind(sim, scene)
env = RaceEnv(cfg, seed=0, quad_factory=IsaacQuad.factory(sim, scene))

ids = torch.arange(env.n, device=env.device)
gc, nrm, _ = env._gate_frame(torch.zeros(env.n, dtype=torch.long, device=env.device))
yaw = env.gate_yaw[:, 0]
env.quad.reset(ids, gc - 3.0 * nrm, yaw)
env.quad.thr[:] = 0.0
rig.update(env)
for _ in range(5):
    sim.step(render=False)  # let the kinematic poses propagate
img = rig.render(0.02)
mx, mn = img.max(1).values, img.min(1).values
sat = (mx - mn) > 0.12  # coloured pixels (background / ground are grey)
r, g_, b_ = img[:, 0], img[:, 1], img[:, 2]
orange = sat & (r > g_) & (g_ > b_) & (r - b_ > 0.2)
frac = orange.float().mean(dim=(1, 2))
print("frame pixel fraction per env:", [round(x, 3) for x in frac.tolist()])
ok1 = bool((frac > 0.01).all())
# marker colour: dominant hue of saturated non-orange pixels must match the token's colour family
fam = {0: 2, 1: 1, 2: 0, 3: None}  # L blue (B), S green (G), R orange (excluded -> use R), U yellow
turn = env.m_turn[ids, 0]
good = 0
for i in range(env.n):
    px = img[i][:, sat[i] & ~orange[i]] if int(turn[i]) != 2 else img[i][:, sat[i] & (r[i] > g_[i]) & (g_[i] > b_[i]) & ~orange[i]]
    if px.shape[1] < 3:
        continue
    m = px.mean(1)
    t = int(turn[i])
    ok = (t == 0 and m[2] > m[0]) or (t == 1 and m[1] > m[0] and m[1] > m[2]) or (t == 2 and m[0] > m[2]) or (t == 3 and m[0] > m[2] and m[1] > m[2])
    good += int(ok)
ok2 = good >= 0.7 * env.n
print("marker colour correct:", good, "/", env.n)
ok3 = bool((img[0] - img[1]).abs().mean() > 1e-3)
before = img.clone()
env.reset_idx(ids)
env.quad.reset(ids, env.gate_pos[:, 0] - 3.0 * torch.stack((torch.cos(env.gate_yaw[:, 0]), torch.sin(env.gate_yaw[:, 0]), torch.zeros_like(yaw)), -1), env.gate_yaw[:, 0])
rig.update(env)
for _ in range(5):
    sim.step(render=False)
after = rig.render(0.02)
ok4 = bool((after - before).abs().mean() > 1e-3)
print(f"PASS frame-visible={ok1} marker-colour={ok2} envs-isolated={ok3} follows-reset={ok4}", flush=True)
try:
    import imageio.v2 as imageio
    tiles = [(img[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8) for i in range(min(8, env.n))]
    imageio.imwrite("/workspace/DroneRacing/runs/vision_check.png", np.concatenate(tiles, 1))
except Exception as e:  # noqa: BLE001
    print("montage failed", e)
app.close()
