"""Isaac Lab (PhysX) rigid-body backend with the same interface as droneracing.quad.QuadBatch.

Our actuator model (thrust lag, rate loop with angular-acceleration limit, drag) runs on the
torch side and is applied to the PhysX rigid body as body-frame force + torque. PhysX provides
rigid-body dynamics (incl. gyroscopic terms); gates/poles stay analytic in RaceEnv, so only
the vehicle physics changes. Quaternions are converted (Isaac xyzw <-> ours wxyz).
"""
from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from droneracing.config import VehicleRanges
from droneracing.quad import GRAVITY, QuadBatch, quat_to_rot, yaw_to_quat


def build_scene(n_envs: int, dt: float, device: str, env_spacing: float = 0.0, extra_cfg=None, ground_size: float | None = None):
    """``extra_cfg(cfg)`` may attach more assets (gates, markers, camera) before the scene is created."""
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg() if ground_size is None else sim_utils.GroundPlaneCfg(size=(ground_size, ground_size)))
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
        quad = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Quad",
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.04, 0.02),  # tiny visual body: the on-board camera must not see the airframe (inertia is set explicitly)
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, enable_gyroscopic_forces=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=device, gravity=(0.0, 0.0, -GRAVITY)))
    cfg = SceneCfg(num_envs=n_envs, env_spacing=env_spacing)
    if extra_cfg is not None:
        extra_cfg(cfg)
    scene = InteractiveScene(cfg)
    sim.reset()
    return sim, scene


class IsaacQuad(QuadBatch):
    def __init__(self, n: int, device: str, cfg: VehicleRanges, gen: torch.Generator, sim=None, scene=None):
        self._ready = False
        self.sim, self.scene = sim, scene
        self.rb = scene["quad"]
        self.body_ids = self.rb.find_bodies(".*")[0]
        super().__init__(n, device, cfg, gen)
        self._ready = True
        self._push_params(torch.arange(n, device=device))

    # make a factory usable by RaceEnv
    @classmethod
    def factory(cls, sim, scene):
        return lambda n, device, cfg, gen: cls(n, device, cfg, gen, sim=sim, scene=scene)

    def _inertia_diag(self, ids):
        i = 0.6 * self.mass[ids] * self.arm[ids] ** 2
        return torch.stack((i, i, 1.6 * i), -1)

    def _push_params(self, ids: torch.Tensor) -> None:
        full = torch.zeros(len(ids), 1, 9, device=self.device)
        d = self._inertia_diag(ids)
        full[:, 0, 0], full[:, 0, 4], full[:, 0, 8] = d[:, 0], d[:, 1], d[:, 2]
        self.rb.set_masses_index(masses=self.mass[ids][:, None].contiguous(), env_ids=ids)
        self.rb.set_inertias_index(inertias=full, env_ids=ids)

    def sample_params(self, ids: torch.Tensor) -> None:
        super().sample_params(ids)
        if self._ready:
            self._push_params(ids)

    def _sync(self) -> None:
        d = self.rb.data
        self.pos = d.root_pos_w.torch.clone() - self.scene.env_origins
        self.vel = d.root_lin_vel_w.torch.clone()
        self.quat = d.root_quat_w.torch[:, [3, 0, 1, 2]].clone()
        self.omega = d.root_ang_vel_b.torch.clone()

    def reset(self, ids: torch.Tensor, pos: torch.Tensor, yaw: torch.Tensor, vel: torch.Tensor | None = None) -> None:
        q_wxyz = yaw_to_quat(yaw)
        pose = torch.cat((pos + self.scene.env_origins[ids], q_wxyz[:, [1, 2, 3, 0]]), -1)
        v6 = torch.zeros(len(ids), 6, device=self.device)
        if vel is not None:
            v6[:, :3] = vel
        self.rb.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.rb.write_root_velocity_to_sim_index(root_velocity=v6, env_ids=ids)
        self.thr[ids] = 1.0 / self.twr[ids]
        # keep torch-side mirrors consistent until the next physics step
        self.pos[ids], self.vel[ids], self.quat[ids], self.omega[ids] = pos, v6[:, :3], q_wxyz, 0.0

    def step(self, action: torch.Tensor, dt: float) -> None:
        thrust, torque_b = self.actuate(action, dt, self.omega)  # motor-level model shared with the torch env
        f_b = torch.zeros(self.n, 3, device=self.device)
        f_b[:, 2] = thrust
        R = quat_to_rot(self.quat)
        f_drag_w = (self.drag_acc_world(R) + self.ext_acc) * self.mass[:, None]
        f_b += (R.transpose(1, 2) @ f_drag_w.unsqueeze(-1)).squeeze(-1)
        self.rb.permanent_wrench_composer.set_forces_and_torques_index(
            body_ids=self.body_ids, forces=f_b[:, None, :], torques=torque_b[:, None, :]
        )
        self.spec_b = f_b / self.mass[:, None]  # accelerometer truth: specific force in the body frame
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt)
        self._sync()
