"""Does the on-board camera prim follow the rigid body after a teleport and after PhysX motion? Prints cam world pos vs quad pos."""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import torch  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/isaac")
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402
from vision_rig import VisionRig  # noqa: E402

n = 4
cfg = make_cfg({"profile": "a2rl", "sensor_scale": 0.0}, n)
rig = VisionRig(n, 96, 128, cam_tilt=0.0)
sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0", env_spacing=150.0, extra_cfg=rig.attach)
rig.bind(sim, scene)
env = RaceEnv(cfg, seed=0, quad_factory=IsaacQuad.factory(sim, scene))
ids = torch.arange(n, device=env.device)


def show(tag):
    rig.update(env)
    img = rig.render(0.02)
    qp = env.quad.pos[0] + rig.origins[0]
    print(f"{tag}: quad_w {[round(float(v), 2) for v in qp]} cam_w {[round(float(v), 2) for v in rig.cam.data.pos_w[0]]} img mean {float(img.mean()):.3f} std {float(img.std()):.3f}", flush=True)


rig.update(env)
show("after env init")
env.quad.reset(ids, torch.tensor([[5.0, 3.0, 2.0]] * n, device=env.device), torch.zeros(n, device=env.device))
for _ in range(5):
    sim.step(render=False)
show("after teleport to (5,3,2) + 5 steps")
env.quad.thr[:] = 0.0
for _ in range(60):
    env.quad.step(torch.zeros(n, 4, device=env.device), cfg.dt)  # free fall under gravity
show("after 60 physics steps (falling)")

def camw(tag):
    print(f"{tag}: cam_w {[round(float(v), 2) for v in rig.cam.data.pos_w[0]]} quad_w {[round(float(v), 2) for v in (env.quad.pos[0] + rig.origins[0])]}", flush=True)


try:
    sim.forward()
    rig.cam.update(dt=0.02)
    camw("A sim.forward() + cam.update")
except Exception as e:  # noqa: BLE001
    print("A failed", e)
try:
    sim.step(render=True)
    rig.cam.update(dt=0.02)
    camw("B sim.step(render=True) + cam.update")
except Exception as e:  # noqa: BLE001
    print("B failed", e)
try:
    pw, qw = rig.cam._view.get_world_poses()
    print("C view.get_world_poses:", [round(float(v), 2) for v in torch.as_tensor(pw.torch if hasattr(pw, "torch") else pw)[0]], flush=True)
except Exception as e:  # noqa: BLE001
    print("C failed", e)
try:
    print("D sim.use_fabric:", getattr(sim.cfg, "use_fabric", None), "physics_sim_view:", type(sim.physics_sim_view).__name__ if hasattr(sim, "physics_sim_view") else None, flush=True)
except Exception as e:  # noqa: BLE001
    print("D failed", e)

try:
    target = rig.origins + torch.tensor([[3.0, 2.0, 1.5]], device=env.device)
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device).repeat(n, 1)
    rig.cam.set_world_poses(positions=target, orientations=q, convention="world")
    pw, _ = rig.cam._view.get_world_poses()
    print("E after set_world_poses (target %s): view says %s" % ([round(float(v), 2) for v in target[0]], [round(float(v), 2) for v in torch.as_tensor(pw.torch)[0]]), flush=True)
    im0 = rig.render(0.02)
    pw2 = rig.cam.data.pos_w
    print("F after render: cam.data.pos_w", [round(float(v), 2) for v in pw2[0]], "img mean %.3f std %.3f" % (float(im0.mean()), float(im0.std())), flush=True)
except Exception as e:  # noqa: BLE001
    import traceback
    print("E failed", e, traceback.format_exc()[-300:])
try:
    st = rig.cam._view
    print("G view type", type(st).__name__, "count", st.count, "prims", [p.GetPath().pathString for p in list(st.prims)[:1]], flush=True)
except Exception as e:  # noqa: BLE001
    print("G failed", e)
app.close()
