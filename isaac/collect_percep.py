"""Collect perception training data in Isaac Lab: RGB frames + ground-truth gate inner corners (with
depth-based visibility), sign token of the target gate, and the true vehicle / gate state.
The vehicle is flown by a state-based teacher (+ exploration noise) so frames follow race-like distributions.

No policy is involved: every frame the vehicle is re-placed at a random pose in front of a random gate of a
freshly drawn track (the env's mid-track reset, widened), so coverage does not depend on any controller.

Run: scripts/run_isaaclab.sh isaac/collect_percep.py --task runs/t2 --n-envs 64 --frames 20000 --out runs/percep/d0
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="run dir whose config.json defines the task / DR (weights unused)")
parser.add_argument("--n-envs", type=int, default=64)
parser.add_argument("--frames", type=int, default=20000)
parser.add_argument("--dist", type=float, nargs=2, default=(1.5, 16.0), help="distance range in front of the gate [m]")
parser.add_argument("--img-h", type=int, default=192)
parser.add_argument("--img-w", type=int, default=256)
parser.add_argument("--hfov", type=float, default=130.0)
parser.add_argument("--tilt", type=float, default=0.0, help="max |roll| [deg] of the random attitude (pitch up to 0.7x)")
parser.add_argument("--yaw-noise", type=float, default=0.0, help="max heading offset from the gate normal [rad]")
parser.add_argument("--lateral", type=float, default=0.0, help="max lateral offset from the gate normal axis [m]")
parser.add_argument("--pitch-range", type=float, nargs=2, default=None, help="nose-down pitch range [deg] (race attitudes); overrides the symmetric --tilt pitch")
parser.add_argument("--roll-max", type=float, default=None, help="max |roll| [deg] when --pitch-range is set")
parser.add_argument("--inverted-frac", type=float, default=0.0, help="fraction of frames with the roll flipped by 180 deg (tumbling / split-S views)")
parser.add_argument("--crop-out", type=int, default=0, help="store adaptive crops (side px) around a noisy prediction of the next gate instead of full frames (MonoRace 384)")
parser.add_argument("--extr-deg", type=float, default=0.0, help="random camera mount error per frame batch, each Euler angle uniform in +-deg")
parser.add_argument("--sync", type=int, default=1, help="1 = rendered physics steps + scene.update after moving the parts (0 = old path: gates stayed at their old place after a reset, 2026-10-22)")
parser.add_argument("--settle", type=int, default=3, help="extra renders after moving gates / vehicle before a frame is kept")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="runs/percep/d0")
parser.add_argument("--preview", type=int, default=0, help="save N annotated preview PNGs")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/isaac")
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402
from vision_rig import VisionRig  # noqa: E402

a = args_cli
torch.manual_seed(a.seed)
root = Path("/workspace/DroneRacing")
tdir = root / a.task
meta = json.loads((tdir / "config.json").read_text())
cfg = make_cfg(dict(meta["args"]), a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
cfg.mid_init_prob = 1.0
cfg.mid_init_dist = tuple(a.dist)
cfg.mid_init_speed = (0.0, 0.0)
rig = VisionRig(a.n_envs, a.img_h, a.img_w, hfov_deg=a.hfov, cam_tilt=cfg.cam_tilt, depth=True)
sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0", env_spacing=150.0, extra_cfg=rig.attach,
                         ground_size=150.0 * (int(a.n_envs ** 0.5) + 2))
rig.bind(sim, scene)
env = RaceEnv(cfg, seed=a.seed, quad_factory=IsaacQuad.factory(sim, scene))
dev = env.device
dt = cfg.dt * cfg.decimation
rig.update(env)
out = root / a.out
out.mkdir(parents=True, exist_ok=True)
imgs, cor, vis, slotv, tok, st = [], [], [], [], [], []
wins = []  # crop windows (x0, y0, side) in full-frame px, for geometry-consistent augmentation (rolling shutter row time, focal length)
got, step, prev_n = 0, 0, 0
with torch.no_grad():
    while got < a.frames:
        ids_all = torch.arange(a.n_envs, device=dev)
        env.reset_idx(ids_all)
        if a.tilt > 0 or a.yaw_noise > 0 or a.lateral > 0 or a.pitch_range is not None:  # attitude / pose diversity beyond the level, centred start
            g_ = torch.minimum(env.gate_idx, env._last_slot())
            gy = env.gate_yaw[ids_all, g_]
            rgt = torch.stack((torch.sin(gy), -torch.cos(gy), torch.zeros_like(gy)), -1)
            pos = env.quad.pos + rgt * ((torch.rand(a.n_envs, 1, device=dev) * 2 - 1) * a.lateral)
            pos[:, 2] = pos[:, 2].clamp_min(0.7)
            yaw = gy + (torch.rand(a.n_envs, device=dev) * 2 - 1) * a.yaw_noise
            roll = (torch.rand(a.n_envs, device=dev) * 2 - 1) * a.tilt * 3.14159 / 180
            pitch = (torch.rand(a.n_envs, device=dev) * 2 - 1) * a.tilt * 0.7 * 3.14159 / 180
            if a.pitch_range is not None:
                lo_, hi_ = a.pitch_range
                pitch = (lo_ + torch.rand(a.n_envs, device=dev) * (hi_ - lo_)) * 3.14159 / 180
                pitch = torch.where(torch.rand(a.n_envs, device=dev) < 0.08, -pitch * 0.5, pitch)  # a few nose-up views (braking)
                roll = (torch.rand(a.n_envs, device=dev) * 2 - 1) * (a.roll_max if a.roll_max is not None else a.tilt) * 3.14159 / 180
                roll = roll + (torch.rand(a.n_envs, device=dev) < a.inverted_frac).float() * 3.14159
            cr, sr, cp, sp, cy_, sy_ = roll.cos() / 1, roll.sin(), pitch.cos(), pitch.sin(), (yaw / 2).cos(), (yaw / 2).sin()
            cr, sr, cp, sp = (roll / 2).cos(), (roll / 2).sin(), (pitch / 2).cos(), (pitch / 2).sin()
            q = torch.stack((cr * cp * cy_ + sr * sp * sy_, sr * cp * cy_ - cr * sp * sy_,
                             cr * sp * cy_ + sr * cp * sy_, cr * cp * sy_ - sr * sp * cy_), -1)
            env.quad.pos, env.quad.quat = pos, q
            pose = torch.cat((pos + env.quad.scene.env_origins, q[:, [1, 2, 3, 0]]), -1)
            env.quad.rb.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids_all)
        if a.extr_deg > 0:
            from cam_artifacts import random_rotation
            rig.extr_err = random_rotation(a.n_envs, a.extr_deg, dev)
        rig.update(env)
        if a.sync == 1:
            for _ in range(3):
                scene.write_data_to_sim()
                rig.sim.step(render=True)
                scene.update(dt)
            rig.update(env)
        else:
            rig.sim.step(render=False)  # kinematic gate poses / vehicle pose are applied by a physics step
        for _ in range(a.settle):  # the renderer lags the physics state by a frame or two
            rig.render(dt)
        rgb = rig.render(dt)  # also refreshes the depth image
        _blank = float((rgb.float().std(dim=(1, 2, 3)) < 0.01).float().mean())
        if _blank > 0.5:  # renderer produced empty frames (seen with 64 envs at 308x410: a whole dataset came out grey)
            n_blank = locals().get("n_blank", 0) + 1
            print(f"blank batch ({_blank:.2f} of frames), skipped", flush=True)
            if n_blank > 20:
                print("Error: renderer keeps returning blank frames; use fewer envs / a smaller image", flush=True)
                break
            step += 1
            continue
        n_blank = 0
        if step > 2:
            px, v, sv = rig.gate_corners(env)
            idx = torch.minimum(env.gate_idx, env._last_slot())
            if a.crop_out > 0:  # window from the true next-gate corners, perturbed like an EKF prediction; 15 % full-frame fallback
                from droneracing.adaptive_crop import crop_windows, extract, to_crop
                co = a.crop_out
                okw = sv[:, 1] & v[:, 1].any(-1) & (torch.rand(a.n_envs, device=dev) > 0.15)
                pu = px[:, 1].clone()
                ctr = pu.mean(1, keepdim=True)
                pu = ctr + (pu - ctr) * (0.8 + 0.6 * torch.rand(a.n_envs, 1, 1, device=dev))
                pu = pu + (torch.rand(a.n_envs, 1, 2, device=dev) * 2 - 1) * 0.15 * (pu.amax(1) - pu.amin(1)).amax(-1).clamp_min(40)[:, None, None]
                win = crop_windows(pu, okw, a.img_h, a.img_w, co)
                rgb_c = extract(rgb, win, co)
                pxc = to_crop(px.reshape(a.n_envs, -1, 2), win, co).reshape(px.shape)
                inside = (pxc[..., 0] >= 0) & (pxc[..., 0] < co) & (pxc[..., 1] >= 0) & (pxc[..., 1] < co)
                imgs.append((rgb_c.clamp(0, 1) * 255).to(torch.uint8).cpu())
                wins.append(win.cpu())
                cor.append((pxc / co).half().cpu())
                vis.append((v & inside).cpu())
            else:
                imgs.append((rgb * 255).to(torch.uint8).cpu())
                cor.append((px / torch.tensor([a.img_w, a.img_h], device=dev)).half().cpu())
                vis.append(v.cpu())
            slotv.append(sv.cpu()); tok.append(env._tokens(idx).cpu())
            st.append(torch.cat((env.quad.pos, env.quad.vel, env.quad.quat, env.quad.omega), -1).cpu())
            got += a.n_envs
            if a.preview and len(imgs) <= a.preview:
                pv_px, pv_w, pv_h = (pxc, a.crop_out, a.crop_out) if a.crop_out > 0 else (px, a.img_w, a.img_h)
                v_pv = vis[-1].to(dev)
                from PIL import Image, ImageDraw
                for k in range(min(4, a.n_envs)):
                    im = Image.fromarray(imgs[-1][k].permute(1, 2, 0).numpy())
                    d = ImageDraw.Draw(im)
                    for s_ in range(4):
                        for c_ in range(4):
                            x, y = pv_px[k, s_, c_].tolist()
                            if bool(v_pv[k, s_, c_]):
                                d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 255, 0), width=2)
                                d.text((x + 4, y - 4), f"{s_}{'tr'[0] if False else 'ABCD'[c_]}", fill=(255, 255, 0))
                            elif bool(sv[k, s_]):
                                if 0 <= x < pv_w and 0 <= y < pv_h:
                                    d.ellipse([x - 2, y - 2, x + 2, y + 2], outline=(255, 60, 60))
                    im.resize((pv_w * 2, pv_h * 2)).save(out / f"preview_{len(imgs):03d}_{k}.png")
        step += 1
        if got - prev_n >= 10 * a.n_envs:
            print(f"frames {got}", flush=True)
            prev_n = got
torch.save({"img": torch.cat(imgs), "corners": torch.cat(cor), "vis": torch.cat(vis), "slot_valid": torch.cat(slotv),
            "tok": torch.cat(tok), "state": torch.cat(st), "hfov": a.hfov, "hw": (a.crop_out, a.crop_out) if a.crop_out > 0 else (a.img_h, a.img_w), "full_hw": (a.img_h, a.img_w), **({"win": torch.cat(wins)} if wins else {})}, out / "data.pt")
print("COLLECT DONE", got, flush=True)
app.close()
