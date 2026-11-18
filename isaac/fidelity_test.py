"""Isaac Lab (PhysX) vs our torch quadrotor model: same params, same action sequence.

Run: scripts/run_isaaclab.sh isaac/fidelity_test.py
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--n-envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app = AppLauncher(args_cli).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
from droneracing.config import VehicleRanges  # noqa: E402
from droneracing.quad import GRAVITY, QuadBatch, quat_to_rot  # noqa: E402

N, DT = args_cli.n_envs, 0.005
dev = "cuda:0"


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
    quad = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Quad",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, enable_gyroscopic_forces=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=dev, gravity=(0.0, 0.0, -GRAVITY)))
scene = InteractiveScene(SceneCfg(num_envs=N, env_spacing=0.0))
sim.reset()
quad = scene["quad"]
body_ids = quad.find_bodies(".*")[0]
ids = torch.arange(N, device=dev)

# reference model + shared parameters
gen = torch.Generator(device=dev)
gen.manual_seed(0)
ref = QuadBatch(N, dev, VehicleRanges(), gen)
gen2 = torch.Generator(device=dev)
gen2.manual_seed(0)
act = QuadBatch(N, dev, VehicleRanges(), gen2)  # identical parameters; owns the Isaac-side actuator state
pos0 = torch.zeros(N, 3, device=dev)
pos0[:, 2] = 5.0
ref.reset(ids, pos0, torch.zeros(N, device=dev))
act.reset(ids, pos0, torch.zeros(N, device=dev))

I_diag = ref.inertia_diag()
inertia_full = torch.zeros(N, 1, 9, device=dev)
inertia_full[:, 0, 0], inertia_full[:, 0, 4], inertia_full[:, 0, 8] = I_diag[:, 0], I_diag[:, 1], I_diag[:, 2]
quad.set_masses_index(masses=ref.mass[:, None].contiguous(), env_ids=ids)
quad.set_inertias_index(inertias=inertia_full, env_ids=ids)

pose = torch.zeros(N, 7, device=dev)
pose[:, 2], pose[:, 6] = 5.0, 1.0  # xyzw identity
quad.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
quad.write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=dev), env_ids=ids)



def action_at(t: float) -> torch.Tensor:
    a = torch.zeros(N, 4, device=dev)
    hover = 2.0 / ref.twr - 1.0
    a[:, 0] = hover + (0.15 if 0.2 < t < 0.6 else 0.0)  # thrust pulse
    a[:, 1] = 0.4 if 0.4 < t < 0.9 else (-0.4 if 0.9 <= t < 1.4 else 0.0)  # roll-rate doublet
    a[:, 2] = 0.3 if 1.0 < t < 1.5 else 0.0  # pitch-rate pulse
    a[:, 3] = 0.3 if 0.5 < t < 1.0 else 0.0  # yaw-rate pulse
    return a


steps = int(2.0 / DT)
report = {}
for k in range(steps):
    t = k * DT
    a = action_at(t).clamp(-1, 1)
    ref.step(a, DT)

    # --- Isaac side
    q_xyzw = quad.data.root_quat_w.torch
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    R = quat_to_rot(q_wxyz)
    v_w = quad.data.root_lin_vel_w.torch
    w_b = quad.data.root_ang_vel_b.torch
    thrust, torque_b = act.actuate(a, DT, w_b)
    f_thrust = torch.zeros(N, 3, device=dev)
    f_thrust[:, 2] = thrust
    f_drag_w = -(v_w * v_w.norm(dim=-1, keepdim=True)) * ref.kd[:, None]
    f_b = f_thrust + (R.transpose(1, 2) @ f_drag_w.unsqueeze(-1)).squeeze(-1)
    quad.permanent_wrench_composer.set_forces_and_torques_index(
        body_ids=body_ids, forces=f_b[:, None, :], torques=torque_b[:, None, :]
    )
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(DT)
    if k + 1 in (int(0.5 / DT), int(1.0 / DT), int(2.0 / DT)):
        err = (quad.data.root_pos_w.torch - ref.pos).norm(dim=-1)
        verr = (quad.data.root_lin_vel_w.torch - ref.vel).norm(dim=-1)
        print(f"t={(k+1)*DT:.1f}s | pos err mean {err.mean():.4f} m  max {err.max():.4f} m | vel err mean {verr.mean():.4f} m/s | "
              f"z isaac {quad.data.root_pos_w.torch[:,2].mean():.3f} torch {ref.pos[:,2].mean():.3f}", flush=True)

print("FIDELITY DONE", flush=True)
app.close()
