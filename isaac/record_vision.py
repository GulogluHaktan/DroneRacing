"""Record the full vision stack in Isaac Lab (no privileged gate map): per row a chase camera (drone marked) and the
onboard FPV frame with GateNet detections drawn. camera -> GateNet -> PnP -> GRU estimator -> RL policy -> CTBR.

Run: scripts/run_isaaclab.sh isaac/record_vision.py --policy runs/t8 --gatenet runs_host/g2/gatenet.pt --estimator runs_host/e4/estimator.pt --out runs/video_vis
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--gatenet", required=True)
parser.add_argument("--estimator", required=True)
parser.add_argument("--out", default="runs/video_vis")
parser.add_argument("--n-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--rows", type=int, nargs="+", default=[1, 5])
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import TiledCameraCfg  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/isaac")
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.estimator import Estimator, Strapdown, det_features, split_out  # noqa: E402
from droneracing.gatenet import GateNet  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402
from droneracing.quad import quat_to_rot  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402
from vision_rig import VisionRig  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

a = args_cli
torch.manual_seed(a.seed)
root = Path("/workspace/DroneRacing")
out = root / a.out
(out / "frames").mkdir(parents=True, exist_ok=True)
for f in (out / "frames").glob("*.jpg"):
    f.unlink()
pdir = root / a.policy
meta = json.loads((pdir / "config.json").read_text())
cfg = make_cfg(dict(meta["args"]), a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
cfg.mid_init_prob = 0.0
CW, CH, IH, IW = 640, 360, 192, 256
rig = VisionRig(a.n_envs, IH, IW, hfov_deg=130.0, cam_tilt=cfg.cam_tilt)


def attach(c):
    rig.attach(c)
    c.chase = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/ChaseCam", update_period=0.0, height=CH, width=CW, data_types=["rgb"],
                             spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.05, 200.0)))


sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0", env_spacing=150.0, extra_cfg=attach,
                         ground_size=150.0 * (int(a.n_envs ** 0.5) + 2))
rig.bind(sim, scene)
chase = scene["chase"]
env = RaceEnv(cfg, seed=a.seed, quad_factory=IsaacQuad.factory(sim, scene))
dev = env.device
ac = ActorCritic(env.policy_dim, env.critic_dim, 4, PPOCfg()).to(dev)
ac.load_state_dict(torch.load(pdir / "model.pt", map_location=dev)["ac"])
ac.eval()
_ck = torch.load(root / a.gatenet, map_location=dev)
gn = GateNet(refine=_ck.get("refine", False)).to(dev)
gn.load_state_dict(_ck["net"])
gn.eval()
ck = torch.load(root / a.estimator, map_location=dev)
est = Estimator(ck.get("hidden", 160)).to(dev)
est.load_state_dict(ck["net"])
est.eval()
n, K = a.n_envs, cfg.obs_history
dt = cfg.dt * cfg.decimation
fx_cam = 14.0 / 20.955 * CW
fx = (IW / 2) / math.tan(math.radians(130.0) / 2)
strap = Strapdown(n, dev)
hist = torch.zeros(n, K, 10, device=dev)
h = None
rig.update(env)
env.compute_obs()
eye_f = None
trail = {k: [] for k in a.rows}
stats = {k: [0, 0, 0] for k in a.rows}
ep_t = {k: 0.0 for k in a.rows}
last = {k: "" for k in a.rows}


def chase_pose():
    global eye_f
    pos, vel = env.quad.pos, env.quad.vel
    sp = vel[:, :2].norm(dim=-1, keepdim=True)
    tgt_dir = torch.where(sp > 1.0, vel[:, :2] / sp.clamp_min(1e-3), torch.tensor([[1.0, 0.0]], device=dev).expand(n, 2))
    cur = tgt_dir if eye_f is None else F.normalize(0.9 * eye_f + 0.1 * tgt_dir, dim=-1)
    eye_f = cur
    eye = pos + torch.cat((-3.4 * cur, torch.full_like(cur[:, :1], 1.4)), -1)
    eye[:, 2].clamp_(min=0.6)
    tgt = pos + torch.cat((1.5 * cur, torch.zeros_like(cur[:, :1])), -1)
    chase.set_world_poses_from_view(eye + scene.env_origins, tgt + scene.env_origins)
    return eye, tgt


def mark(img, k, eye, tgt):
    f = F.normalize(tgt[k] - eye[k], dim=0)
    r = F.normalize(torch.linalg.cross(f, torch.tensor([0.0, 0.0, 1.0], device=dev)), dim=0)
    u = torch.linalg.cross(r, f)

    def proj(p):
        d = p - eye[k]
        z = float(d @ f)
        return (CW / 2 + fx_cam * float(d @ r) / z, CH / 2 - fx_cam * float(d @ u) / z) if z > 0.2 else None

    trail[k].append(env.quad.pos[k].clone())
    del trail[k][:-40]
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    pts = [p for p in (proj(q) for q in trail[k]) if p is not None]
    if len(pts) > 1:
        d.line(pts, fill=(255, 160, 0), width=2)
    if pts:
        x, y = pts[-1]
        d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(255, 60, 0), width=3)
    return im


def label(im, text):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(im)


with torch.no_grad():
    for step in range(a.steps):
        eye, tgt = chase_pose()
        rgb = rig.render(dt)
        chase.update(dt=dt)
        om, v = env.quad.omega, env.quad.vel
        expo = 0.002 + 0.004 * torch.rand(n, device=dev)
        sgn = torch.where(torch.rand(n, 2, device=dev) < 0.5, -1.0, 1.0)
        flow = torch.stack((fx * expo * (om[:, 2].abs() + v.norm(dim=-1) / 6.0), fx * expo * om[:, 1].abs()), -1) * sgn
        img = realistic_augment(rgb, 0.6, flow_px=flow)  # mild: keep the real render readable in the video
        dets = gn.detect(img)
        feat = det_features(dets, IH, IW, dev, cam_tilt=cfg.cam_tilt)
        imu = env.imu_obs()
        o, h = est(torch.cat((feat, imu, env.prev_action, strap.step(imu)), -1)[:, None], h)
        vel, up, rel, nrm, seen_l = split_out(o[:, 0])
        flag = (seen_l > 0).float()[..., None]
        parts = [vel, up, imu[:, :3], env.prev_action]
        for j in range(2):
            parts += [rel[:, j] * flag[:, j], nrm[:, j] * flag[:, j], env.gate_size[:, None] * flag[:, j], flag[:, j]]
        pol_obs = torch.cat(parts + [hist.reshape(n, -1)], -1)
        hist = torch.roll(hist, -1, dims=1)
        hist[:, -1] = torch.cat((vel, imu[:, :3], env.prev_action), -1)
        Rw = quat_to_rot(env.quad.quat)
        vb = (Rw.transpose(1, 2) @ env.quad.vel.unsqueeze(-1)).squeeze(-1) / 10.0
        rows = []
        for k in a.rows:
            c = chase.data.output["rgb"][k, ..., :3].clamp(0, 255).byte().cpu().numpy()
            cim = mark(c, k, eye, tgt)
            fim = Image.fromarray((img[k].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()).resize((480, 360), Image.NEAREST)
            dr = ImageDraw.Draw(fim)
            for dd in dets[k]:
                p = (dd[1:].view(4, 2) * (480 / IW)).tolist()
                dr.polygon([tuple(q) for q in p], outline=(255, 255, 0))
            dr.text((6, 340), f"estimator: gate0 {'KNOWN' if flag[k, 0, 0] > 0 else 'unseen'}  gate1 {'KNOWN' if flag[k, 1, 0] > 0 else 'unseen'}", fill=(0, 255, 255))
            spd = float(env.quad.vel[k].norm())
            fi, cr, to = stats[k]
            txt = (f"VISION STACK (camera+IMU only)  gate {int(env.gate_idx[k])}/{int(env.n_gates[k])}  v {spd:4.1f} m/s  t {ep_t[k]:4.1f}s  "
                   f"| {fi} ok / {cr} crash / {to} timeout  {last[k]}")
            both = np.concatenate((np.asarray(cim), np.asarray(fim)), 1)
            rows.append(label(Image.fromarray(both), txt))
        imageio.imwrite(out / "frames" / f"{step:05d}.jpg", np.concatenate(rows, 0), quality=88)
        act = ac.actor(ac.pn(pol_obs)).clamp(-1, 1)
        _, _, _, _, done, info = env.step(act)
        rig.update(env)
        if done.any():
            h = h * (~done).float()[None, :, None]
            hist = hist * (~done).float()[:, None, None]
            strap.reset(done)
        for k in a.rows:
            ep_t[k] += dt
            if bool(done[k]):
                fin, cr = bool(info["finished"][k]), bool(info["crashed"][k])
                stats[k][0 if fin else (1 if cr else 2)] += 1
                last[k] = f"last: {'FINISHED' if fin else ('CRASHED' if cr else 'TIMEOUT')} {int(info['gates_passed'][k])}/{int(info['n_gates'][k])} in {ep_t[k]:.1f}s"
                ep_t[k] = 0.0
                trail[k].clear()
print("VIDEO FRAMES DONE", step + 1, flush=True)
app.close()
