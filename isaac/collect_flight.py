"""Closed-loop flight data for the estimator: the state policy flies (its own analytic camera-limited observation),
while every step an Isaac frame is rendered, augmented (backgrounds, blur, noise), passed through GateNet, and the
detections + IMU + previous command are logged together with the true state / gate labels.

Run: scripts/run_isaaclab.sh isaac/collect_flight.py --task runs/t3 --gatenet runs_host/g1/gatenet.pt --n-envs 48 --steps 3000 --out runs/flight/f0
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="policy run dir (config.json + model.pt)")
parser.add_argument("--gatenet", type=str, required=True)
parser.add_argument("--n-envs", type=int, default=48)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--aug", type=float, default=1.0)
parser.add_argument("--hfov", type=float, default=130.0)
parser.add_argument("--img-h", type=int, default=192)
parser.add_argument("--img-w", type=int, default=256)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", type=str, default="runs/flight/f0")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/isaac")
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.estimator import det_features  # noqa: E402
from droneracing.gatenet import GateNet, decode  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402
from droneracing.quad import quat_to_rot  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402
from vision_rig import VisionRig  # noqa: E402

a = args_cli
torch.manual_seed(a.seed)
root = Path("/workspace/DroneRacing")
tdir = root / a.task
meta = json.loads((tdir / "config.json").read_text())
cfg = make_cfg(dict(meta["args"]), a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
cfg.sensors.scale = 1.0
rig = VisionRig(a.n_envs, a.img_h, a.img_w, hfov_deg=a.hfov, cam_tilt=cfg.cam_tilt)
sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0", env_spacing=150.0, extra_cfg=rig.attach,
                         ground_size=150.0 * (int(a.n_envs ** 0.5) + 2))
rig.bind(sim, scene)
env = RaceEnv(cfg, seed=a.seed, quad_factory=IsaacQuad.factory(sim, scene))
dev = env.device
ac = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg()).to(dev)
ac.load_state_dict(torch.load(tdir / "model.pt", map_location=dev)["ac"])
ac.eval()
_ck = torch.load(root / a.gatenet, map_location=dev)
gn = GateNet(refine=_ck.get("refine", False)).to(dev)
gn.load_state_dict(_ck["net"])
gn.eval()
dt = cfg.dt * cfg.decimation
H, W = a.img_h, a.img_w
fx = (W / 2) / math.tan(math.radians(a.hfov) / 2)
n = a.n_envs
rig.update(env)
obs, _ = env.compute_obs()
buf = {k: [] for k in ("feat", "imu", "act", "vel", "up", "gate", "done", "valid")}
since_reset = torch.zeros(n, dtype=torch.long, device=dev)
with torch.no_grad():
    for step in range(a.steps):
        rgb = rig.render(dt)
        om, v = env.quad.omega, env.quad.vel
        expo = 0.002 + 0.004 * torch.rand(n, device=dev)
        sgn = torch.where(torch.rand(n, 2, device=dev) < 0.5, -1.0, 1.0)
        flow = torch.stack((fx * expo * (om[:, 2].abs() + v.norm(dim=-1) / 6.0), fx * expo * om[:, 1].abs()), -1) * sgn
        img = realistic_augment(rgb, a.aug, flow_px=flow)
        feat = det_features(gn.detect(img), H, W, dev, cam_tilt=cfg.cam_tilt)
        imu = env.imu_obs()
        prev = env.prev_action.clone()
        R = quat_to_rot(env.quad.quat)
        Rt = R.transpose(1, 2)
        vel_b = (Rt @ env.quad.vel.unsqueeze(-1)).squeeze(-1) / 10.0
        up_b = R[:, 2, :]
        last = env._last_slot()
        gl = []
        for j in range(2):
            idx = torch.minimum(env.gate_idx + j, last)
            c, nrm, _ = env._gate_frame(idx)
            rel = (Rt @ (c - env.quad.pos).unsqueeze(-1)).squeeze(-1) / 5.0
            nb = (Rt @ nrm.unsqueeze(-1)).squeeze(-1)
            gl.append(torch.cat((rel, nb, env.vis_seen[env.arange, idx].float()[:, None]), -1))
        for k, val in zip(buf, (feat, imu, prev, vel_b, up_b, torch.stack(gl, 1))):
            buf[k].append(val.cpu())
        buf["valid"].append((since_reset >= 3).cpu())
        act = ac.actor(ac.pn(obs))
        obs, _, _, _, done, _ = env.step(act)
        rig.update(env)
        buf["done"].append(done.cpu())
        since_reset = torch.where(done, torch.zeros_like(since_reset), since_reset + 1)
        if step % 200 == 0:
            print(f"step {step}", flush=True)
out = root / a.out
out.mkdir(parents=True, exist_ok=True)
torch.save({k: torch.stack(v) for k, v in buf.items()}, out / "data.pt")
print("FLIGHT DONE", a.steps * n, flush=True)
app.close()
