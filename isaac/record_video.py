"""Record a demo video in Isaac Lab: privileged teacher (top row) vs. RGB+IMU student (bottom row).
Each row = third-person chase camera + the 48x64 student camera (upscaled). Frames are written as JPEGs;
compose with ffmpeg on the host.

Run: scripts/run_isaaclab.sh isaac/record_video.py --teacher runs/t1 --student runs/s2/student.pt --out runs/video
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--teacher", type=str, required=True)
parser.add_argument("--student", type=str, default="")  # empty: both rows are the state policy
parser.add_argument("--out", type=str, default="runs/video")
parser.add_argument("--n-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--img-h", type=int, default=48)
parser.add_argument("--img-w", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.sensors.camera import TiledCameraCfg  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/isaac")
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402
from droneracing.student import StudentNet  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402
from vision_rig import VisionRig  # noqa: E402

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = ImageDraw = None

a = args_cli
root = Path("/workspace/DroneRacing")
out = root / a.out
(out / "frames").mkdir(parents=True, exist_ok=True)
for f in (out / "frames").glob("*.jpg"):
    f.unlink()
tdir = root / a.teacher
meta = json.loads((tdir / "config.json").read_text())
cfg = make_cfg(dict(meta["args"]), a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
cfg.mid_init_prob = 0.0  # full laps from the start line
torch.manual_seed(a.seed)

CW, CH = 640, 360
rig = VisionRig(a.n_envs, a.img_h, a.img_w, cam_tilt=cfg.cam_tilt)


def attach(c):
    rig.attach(c)
    c.chase = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/ChaseCam", update_period=0.0, height=CH, width=CW, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.05, 150.0)),
    )


sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0", env_spacing=150.0, extra_cfg=attach, ground_size=2000.0)
rig.bind(sim, scene)
chase = scene["chase"]
env = RaceEnv(cfg, seed=a.seed, quad_factory=IsaacQuad.factory(sim, scene))
dev = env.device
teacher = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg()).to(dev)
teacher.load_state_dict(torch.load(tdir / "model.pt", map_location=dev)["ac"])
teacher.eval()
student = StudentNet(img_hw=(a.img_h, a.img_w)).to(dev)
if a.student:
    student.load_state_dict(torch.load(root / a.student, map_location=dev)["student"])
student.eval()

n = a.n_envs
half = n // 2
T_ENV, S_ENV = 1, half + 1  # recorded envs: teacher, student (env 0 sits on the world-origin ground and renders washed out)
is_teacher = torch.arange(n, device=dev) < half
dt = cfg.dt * cfg.decimation
rig.update(env)
h = torch.zeros(1, n, student.hidden, device=dev)
eye_f = None
ep_t = {T_ENV: 0.0, S_ENV: 0.0}
last = {T_ENV: "", S_ENV: ""}
stats = {T_ENV: [0, 0, 0], S_ENV: [0, 0, 0]}  # finished, crashed, timeout


def chase_pose():
    global eye_f
    pos, vel = env.quad.pos, env.quad.vel
    sp = vel[:, :2].norm(dim=-1, keepdim=True)
    yaw_dir = torch.where(sp > 1.0, vel[:, :2] / sp.clamp_min(1e-3), torch.zeros_like(vel[:, :2]))
    tgt_dir = torch.where(sp > 1.0, yaw_dir, torch.tensor([[1.0, 0.0]], device=dev).expand_as(yaw_dir))
    cur = tgt_dir if eye_f is None else F.normalize(0.9 * eye_f + 0.1 * tgt_dir, dim=-1)
    eye_f = cur
    eye = pos + torch.cat((-3.2 * cur, torch.full_like(cur[:, :1], 1.3)), -1)
    eye[:, 2].clamp_(min=0.6)
    tgt = pos + torch.cat((1.5 * cur, torch.zeros_like(cur[:, :1])), -1)
    global EYE, TGT
    EYE, TGT = eye, tgt
    chase.set_world_poses_from_view(eye + scene.env_origins, tgt + scene.env_origins)


EYE = TGT = None
TRAIL = {}


def mark_drone(img, k):
    # the airframe mesh is tiny on purpose (on-board camera must not see it); overlay its position + trail
    if Image is None:
        return img
    f = F.normalize(TGT[k] - EYE[k], dim=0)
    r = F.normalize(torch.linalg.cross(f, torch.tensor([0.0, 0.0, 1.0], device=dev)), dim=0)
    u = torch.linalg.cross(r, f)
    fx = 14.0 / 20.955 * CW

    def proj(p):
        d = p - EYE[k]
        z = float(d @ f)
        return (CW / 2 + fx * float(d @ r) / z, CH / 2 - fx * float(d @ u) / z) if z > 0.2 else None

    tr = TRAIL.setdefault(k, [])
    tr.append(env.quad.pos[k].clone())
    del tr[:-40]
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    pts = [proj(p) for p in tr]
    pts = [p for p in pts if p is not None]
    if len(pts) > 1:
        d.line(pts, fill=(255, 160, 0), width=2)
    if pts:
        x, y = pts[-1]
        d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(255, 60, 0), width=3)
        d.line([x - 14, y, x + 14, y], fill=(255, 60, 0), width=2)
        d.line([x, y - 14, x, y + 14], fill=(255, 60, 0), width=2)
    return np.asarray(im)


def to_np(x):
    return x.clamp(0, 255).byte().cpu().numpy()


def label(img, text):
    if Image is None:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(im)


with torch.no_grad():
    for step in range(a.steps):
        raw = None
        chase_pose()
        raw = rig.render(dt)
        chase.update(dt=dt)
        prop = torch.cat((env.imu_obs(), env.prev_action), -1)
        t_act = teacher.actor(teacher.pn(env.teacher_obs())).clamp(-1, 1)
        s_act, _, h = student(raw[:, None], prop[:, None], h)
        act = torch.where(is_teacher[:, None], t_act, s_act[:, 0]) if a.student else t_act
        # ---- compose frame ----
        rows = []
        for k, name in ((T_ENV, "TEACHER  (privileged state)" if a.student else "STATE POLICY  (RL, PPO)"), (S_ENV, "STUDENT  (RGB 48x64 + IMU only)" if a.student else "STATE POLICY  (second track)")):
            c = chase.data.output["rgb"][k, ..., :3].float()
            f = raw[k].permute(1, 2, 0) * 255.0
            f = F.interpolate(f.permute(2, 0, 1)[None], size=(CH, 480), mode="nearest")[0].permute(1, 2, 0)
            spd = float(env.quad.vel[k].norm())
            fi, cr, to = stats[k]
            txt = (f"{name}   gate {int(env.gate_idx[k])}/{int(env.n_gates[k])}   v {spd:4.1f} m/s   t {ep_t[k]:4.1f}s   "
                   f"| laps: {fi} ok / {cr} crash / {to} timeout   last: {last[k]}")
            rows.append(label(np.concatenate((mark_drone(to_np(c), k), to_np(f)), 1), txt))
        imageio.imwrite(out / "frames" / f"{step:05d}.jpg", np.concatenate(rows, 0), quality=88)
        # ---- step ----
        _, _, _, _, done, info = env.step(act)
        rig.update(env)
        h = h * (~done).float()[None, :, None]
        for k in (T_ENV, S_ENV):
            ep_t[k] += dt
            if bool(done[k]):
                fin, cr = bool(info["finished"][k]), bool(info["crashed"][k])
                stats[k][0 if fin else (1 if cr else 2)] += 1
                last[k] = f"{'FINISHED' if fin else ('CRASHED' if cr else 'TIMEOUT')} {int(info['gates_passed'][k])}/{int(info['n_gates'][k])} gates in {ep_t[k]:.1f}s"
                print(f"env {k}: {last[k]}", flush=True)
                ep_t[k] = 0.0
print("VIDEO FRAMES DONE", step + 1, flush=True)
app.close()
