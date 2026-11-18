"""Render the 5 hand-authored test scenes in Isaac Sim (overview / side / top / FPV) as PNGs.

Run: scripts/run_isaaclab.sh isaac/render_scenes.py --enable_cameras
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=str, default="/workspace/DroneRacing/runs/scenes")
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--dome", type=float, default=350.0)
parser.add_argument("--sun", type=float, default=900.0)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import os  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
from droneracing.scenes import ALL_SCENES, scene_prims  # noqa: E402

os.makedirs(args_cli.out_dir, exist_ok=True)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World/Scenes")


def set_matrix(prim_geom, center, R, scale):
    m = Gf.Matrix4d(1.0)
    for i in range(3):
        row = R[:, i] * scale[i]
        m.SetRow(i, Gf.Vec4d(float(row[0]), float(row[1]), float(row[2]), 0.0))
    m.SetRow(3, Gf.Vec4d(float(center[0]), float(center[1]), float(center[2]), 1.0))
    UsdGeom.Xformable(prim_geom).AddTransformOp().Set(m)


COLOR_GAIN = 0.55  # RTX default exposure washes colours out; darken base colours


def draw(prims, root):
    for i, p in enumerate(prims):
        path = f"{root}/p{i:04d}"
        if p.kind == "box":
            g = UsdGeom.Cube.Define(stage, path)
            g.GetSizeAttr().Set(1.0)
            set_matrix(g, p.center, p.R, p.size)
        else:
            g = UsdGeom.Cylinder.Define(stage, path)
            g.GetRadiusAttr().Set(1.0)
            g.GetHeightAttr().Set(1.0)
            g.GetAxisAttr().Set("Z")
            set_matrix(g, p.center, p.R, (p.size[0], p.size[0], p.size[1]))
        g.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) * COLOR_GAIN for c in p.color])])


def ground_and_grid(x0, x1, y0, y1, yoff):
    from droneracing.scenes import box
    prims = [box([(x0 + x1) / 2, (y0 + y1) / 2 + yoff, -0.05], (x1 - x0 + 40, y1 - y0 + 40, 0.1), None, (0.10, 0.14, 0.11))]
    for x in np.arange(np.floor(x0 / 2) * 2, x1 + 1, 2.0):
        prims.append(box([x, (y0 + y1) / 2 + yoff, 0.004], (0.03, y1 - y0, 0.005), None, (0.34, 0.42, 0.37)))
    for y in np.arange(np.floor(y0 / 2) * 2, y1 + 1, 2.0):
        prims.append(box([(x0 + x1) / 2, y + yoff, 0.004], (x1 - x0, 0.03, 0.005), None, (0.34, 0.42, 0.37)))
    return prims


# lighting
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(args_cli.dome)
dome.CreateColorAttr(Gf.Vec3f(0.75, 0.85, 1.0))
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(args_cli.sun)
sun.CreateAngleAttr(0.6)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 15.0, 35.0))

SPACING = 220.0
cams, meta = [], []
for i, make in enumerate(ALL_SCENES):
    sc = make()
    yoff = i * SPACING
    prims = scene_prims(sc, offset=(0.0, yoff, 0.0))
    pts = np.array([p.center for p in prims])
    lo, hi = pts.min(0), pts.max(0)
    draw(ground_and_grid(lo[0] - 8, hi[0] + 8, lo[1] - yoff - 8, hi[1] - yoff + 8, yoff), f"/World/Scenes/ground{i}")
    UsdGeom.Xform.Define(stage, f"/World/Scenes/s{i}")
    draw(prims, f"/World/Scenes/s{i}")
    c = (lo + hi) / 2
    meta.append((sc, c, hi - lo, np.array(sc.start) + np.array([0, yoff, 0])))


def make_cam(path, focal):
    cfg = CameraCfg(
        prim_path=path, update_period=0, height=args_cli.height, width=args_cli.width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=20.955, clipping_range=(0.05, 400.0)),
    )
    return Camera(cfg=cfg)


cam_view = make_cam("/World/CamView", 14.0)
cam_fpv = make_cam("/World/CamFpv", 9.0)
sim.reset()
for _ in range(90):
    sim.step(render=True)
    cam_view.update(dt=sim.get_physics_dt())
    cam_fpv.update(dt=sim.get_physics_dt())


def shoot(cam, eye, target, name):
    e = torch.tensor([eye], device=sim.device, dtype=torch.float32)
    t = torch.tensor([target], device=sim.device, dtype=torch.float32)
    print("shoot", name, [round(float(v),1) for v in eye], [round(float(v),1) for v in target], flush=True)
    cam.set_world_poses_from_view(e, t)
    for _ in range(24):
        sim.step(render=True)
        cam.update(dt=sim.get_physics_dt())
    rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3]
    imageio.imwrite(os.path.join(args_cli.out_dir, name), np.clip(rgb, 0, 255).astype(np.uint8))
    print("saved", name, flush=True)


for sc, c, ext, start in meta:
    dist = 0.62 * max(ext[0], ext[1]) + 4.0
    yoff = start[1] - sc.start[1]
    shoot(cam_view, c + np.array([-0.45 * dist, -0.9 * dist, 0.5 * dist]), c, f"{sc.key}_overview.png")
    shoot(cam_view, c + np.array([0.0, -0.95 * dist, 0.05 * ext[2]]), c, f"{sc.key}_side.png")
    shoot(cam_view, c + np.array([-0.01, 0.0, 1.0 * dist]), c, f"{sc.key}_top.png")
    f = np.array(sc.focus) + np.array([0.0, yoff, 0.0])
    d = sc.focus_dist
    shoot(cam_view, f + np.array([-0.4 * d, -0.9 * d, 0.35 * d]), f, f"{sc.key}_close.png")
    g0 = sc.gates[0].pos
    shoot(cam_fpv, start, np.array([g0[0], g0[1] + yoff, g0[2]]), f"{sc.key}_fpv.png")

print("RENDER DONE", flush=True)
app.close()
