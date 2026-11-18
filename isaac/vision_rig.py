"""Isaac scene content for vision training: gates, markers, poles, landing pad and a body-mounted
TiledCamera, per environment, driven by RaceEnv state.

Everything lives in Isaac (PhysX world, RTX renderer). RaceEnv keeps working in env-local coordinates;
env i sits at ``scene.env_origins[i]`` (>= 150 m apart so cameras never see other tracks).

Pool design (kinematic cuboids, one RigidObject *view* per part, batched over envs):
for each window slot (previous gate, current, next, next-next): 4 frame bars, 4 turn panels
(L/S/R/U, only one shown), finish panel, 3 climb stripes (only one shown), fork panel + side square,
and 4 decoy bars. Unused parts are parked below the ground. 6 poles + a landing pad per env.
Gate colour / lighting randomization can be added later via material variants.

NOT TESTED YET (written while the GPU was busy) - run isaac/check_vision_rig.py first.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.camera import TiledCameraCfg

# A2RL x DCL gate (Novak et al. 2602.01860; Bahnam et al. 2601.15222): square, 1.5 m inner opening, 2.7 m outer
# -> 0.6 m wide printed panel frame, purple with white A2RL / DCL logos, thin violet LED strip on the opening edge
BAR, DEPTH, FRAME_LEN, SIDE_LEN = 0.60, 0.10, 2.7, 1.5
LED_RGB = (0.75, 0.45, 1.0)
LOGO = ((-0.6, 0.0), (0.6, 0.0))  # lateral offsets of the two logos on the bottom bar
TURN_RGB = [(0.10, 0.45, 1.00), (0.10, 0.85, 0.30), (1.00, 0.55, 0.10), (0.95, 0.90, 0.15)]
CLIMB_RGB = [(0.05, 0.05, 0.05), (0.50, 0.50, 0.50), (1.00, 1.00, 1.00)]
# gate_racer / Swift look: dark navy frame + white corner squares (the detector keypoints); neutral poles
GATE_RGB, FINISH_RGB, FORK_RGB, WHITE = (0.30, 0.20, 0.70), (1.0, 0.1, 0.8), (0.16, 0.16, 0.20), (1.0, 1.0, 1.0)
POLE_RGB = (0.55, 0.57, 0.62)
PARK_Z = -50.0
JITTER = (0.35, 0.12, 0.10)  # sign lateral shift [m], in-plane tilt [rad], yaw wobble [rad]
PANEL_DARK = (0.04, 0.04, 0.05)
PATCH_SIZE = [(DEPTH + 0.05, 0.16, 0.34), (DEPTH + 0.05, 0.16, 0.34), (DEPTH + 0.05, 0.16, 0.34), (DEPTH + 0.05, 0.52, 0.08)]  # left, centre, right, bottom
PATCH_POS = [(-0.18, 0.02), (0.0, 0.02), (0.18, 0.02), (0.0, -0.18)]  # (lateral, vertical) offsets on the panel
PATCH_SET = {0: (0,), 1: (1,), 2: (2,), 3: (0, 2, 3), 4: (0, 1, 2, 3)}  # class (4 = finish) -> patches
COLOR_GAIN = 0.55  # RTX default exposure washes colours out (same trick as render_scenes.py)


def _g(rgb):
    return tuple(float(c) * COLOR_GAIN for c in rgb)


def _cuboid(size, rgb, emissive=None):
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=_g(rgb), **({} if emissive is None else {"emissive_color": tuple(float(c) for c in emissive)})),
    )


def _cyl(radius, height, rgb):
    return sim_utils.CylinderCfg(
        radius=radius, height=height,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_g(rgb)),
    )


# Visual-only racing-drone body (A2RL-style: navy/violet fuselage, orange canopy, X arms with amber LED strips, motors, props).
# (name, kind, size, local pos [m], local yaw [rad], rgb, emissive). Body frame: x forward, y left, z up. Off by default
# (VisionRig.drone_model): the on-board camera must not see the airframe in evaluations; use it for chase-view videos.
_AR = 0.115 / math.sqrt(2)  # motor offset along x and y (arm = 0.115 m centre-to-motor)
_DM = [("body", "box", (0.15, 0.07, 0.035), (-0.005, 0.0, 0.0), 0.0, (0.12, 0.12, 0.45), None),
       ("canopy", "box", (0.085, 0.05, 0.03), (-0.01, 0.0, 0.028), 0.0, (1.0, 0.45, 0.05), None),
       ("stripe", "box", (0.10, 0.012, 0.037), (-0.005, 0.0, 0.0), 0.0, (0.85, 0.85, 0.9), None),
       ("pod", "box", (0.03, 0.045, 0.03), (0.07, 0.0, -0.002), 0.0, (0.05, 0.05, 0.06), None),
       ("batt", "box", (0.08, 0.04, 0.03), (-0.01, 0.0, -0.032), 0.0, (0.08, 0.08, 0.09), None)]
for _i, (_sx, _sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
    _yaw = math.atan2(_sy, _sx)
    _DM += [(f"arm{_i}", "box", (0.12, 0.018, 0.010), (_sx * _AR / 2, _sy * _AR / 2, 0.0), _yaw, (0.10, 0.10, 0.12), None),
            (f"led{_i}", "box", (0.11, 0.006, 0.004), (_sx * _AR / 2, _sy * _AR / 2, 0.007), _yaw, (1.0, 0.65, 0.1), (1.0, 0.6, 0.1)),
            (f"mot{_i}", "cyl", (0.015, 0.028), (_sx * _AR, _sy * _AR, 0.012), 0.0, (0.15, 0.15, 0.17), None),
            (f"prop{_i}", "cyl", (0.065, 0.002), (_sx * _AR, _sy * _AR, 0.028), 0.0, (0.3, 0.3, 0.33), None)]


# opponent drone (visual only, flat, yaw only): orange body + crossed arms + props; same size as ours
_OM = [("body", "box", (0.15, 0.07, 0.035), (0.0, 0.0, 0.0), 0.0, (1.0, 0.35, 0.05), (0.6, 0.15, 0.0)),
       ("armA", "box", (0.26, 0.02, 0.012), (0.0, 0.0, 0.0), math.pi / 4, (0.08, 0.08, 0.10), None),
       ("armB", "box", (0.26, 0.02, 0.012), (0.0, 0.0, 0.0), -math.pi / 4, (0.08, 0.08, 0.10), None)]
for _i, (_sx, _sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
    _OM.append((f"prop{_i}", "cyl", (0.065, 0.004), (_sx * _AR, _sy * _AR, 0.02), 0.0, (0.35, 0.35, 0.38), None))


def _mat_to_quat_xyzw(m: np.ndarray) -> tuple:
    w = math.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = (m[2, 1] - m[1, 2]) / (4 * w)
    y = (m[0, 2] - m[2, 0]) / (4 * w)
    z = (m[1, 0] - m[0, 1]) / (4 * w)
    return (x, y, z, w)  # this Isaac Lab build uses xyzw quaternions (offset.rot of the camera included)


def _rot_to_quat_xyzw_torch(R: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrices [N,3,3] -> quaternions [N,4] (x, y, z, w), w >= 0."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    t = m00 + m11 + m22
    w = 0.5 * torch.sqrt((1.0 + t).clamp_min(1e-9))
    x = 0.5 * torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(1e-9))
    y = 0.5 * torch.sqrt((1.0 - m00 + m11 - m22).clamp_min(1e-9))
    z = 0.5 * torch.sqrt((1.0 - m00 - m11 + m22).clamp_min(1e-9))
    x = torch.copysign(x, R[:, 2, 1] - R[:, 1, 2])
    y = torch.copysign(y, R[:, 0, 2] - R[:, 2, 0])
    z = torch.copysign(z, R[:, 1, 0] - R[:, 0, 1])
    return torch.stack((x, y, z, w), -1)


class VisionRig:
    WINDOW = (-1, 0, 1, 2)  # gate slots relative to the target gate

    def __init__(self, n_envs: int, height: int = 48, width: int = 64, hfov_deg: float = 120.0,
                 cam_tilt: float = 0.75, n_poles: int = 6, depth: bool = False):
        self.n, self.H, self.W, self.cam_tilt, self.n_poles = n_envs, height, width, cam_tilt, n_poles
        self.depth = depth
        self.hfov = hfov_deg
        self.opp_k = 0  # number of opponent drones rendered per env (set before attach(); = cfg.opp_max)
        self.drone_model = False  # set True before attach(): add the visual drone body (chase-view videos only)
        self.extr_err = None  # optional [N,3,3] camera-frame rotation: true mount differs from the nominal one (eval_vision --tilt-err)
        self.objs: dict[str, RigidObjectCfg] = {}
        self.parts: list[str] = []

    # ---- scene description (passed to build_scene(extra_cfg=rig.attach)) --------------------
    def _add(self, name, spawn):
        self.objs[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PARK_Z)),
        )

    def attach(self, cfg) -> None:
        for w, rel in enumerate(self.WINDOW):
            p = f"s{w}"
            for b in range(4):
                size = (DEPTH, FRAME_LEN, BAR) if b < 2 else (DEPTH, BAR, SIDE_LEN)
                self._add(f"{p}_bar{b}", _cuboid(size, GATE_RGB))
                self._add(f"{p}_dbar{b}", _cuboid(size, GATE_RGB))  # decoy gate (same look)
                lsz = (DEPTH + 0.02, 0.5, 0.14) if b < 2 else (DEPTH + 0.02, 0.14, 0.5)  # logos: 2 on the bottom bar, 1 per side bar
                self._add(f"{p}_cor{b}", _cuboid(lsz, WHITE))
                self._add(f"{p}_dcor{b}", _cuboid(lsz, WHITE))
                lsz = (DEPTH + 0.04, SIDE_LEN, 0.05) if b < 2 else (DEPTH + 0.04, 0.05, SIDE_LEN)  # LED strip on the opening edge
                self._add(f"{p}_led{b}", _cuboid(lsz, LED_RGB, emissive=LED_RGB))
                self._add(f"{p}_dled{b}", _cuboid(lsz, LED_RGB, emissive=LED_RGB))
            self._add(f"{p}_post", _cuboid((0.08, 0.08, 3.5), (0.12, 0.12, 0.14)))
            self._add(f"{p}_dpost", _cuboid((0.08, 0.08, 3.5), (0.12, 0.12, 0.14)))
            # monochrome sign: dark base + white patches (shape code, no colour): L=left, S=centre,
            # R=right, U=left+right+bottom, finish=all four
            self._add(f"{p}_pbase", _cuboid((DEPTH + 0.03, 0.55, 0.45), PANEL_DARK))  # printed on the top bar
            for k_, sz in enumerate(PATCH_SIZE):
                self._add(f"{p}_patch{k_}", _cuboid(sz, WHITE))
            for c, rgb in enumerate(CLIMB_RGB):
                self._add(f"{p}_climb{c}", _cuboid((DEPTH + 0.03, 0.50, 0.12), rgb))  # climb code printed on the bottom bar
            self._add(f"{p}_fork", _cuboid((DEPTH + 0.03, 0.4, 0.4), FORK_RGB))  # fork sign printed on the top bar, off to one side
            self._add(f"{p}_forksq", _cuboid((DEPTH + 0.05, 0.16, 0.16), WHITE))
        for k in range(self.n_poles):
            self._add(f"pole{k}", _cyl(0.15, 3.5, POLE_RGB))
        self._add("pad", _cyl(1.5, 0.04, (0.95, 0.95, 0.95)))
        if self.drone_model:
            for nm, kind, sz, _pos, _yaw, rgb, em in _DM:
                self._add(f"dm_{nm}", _cuboid(sz, rgb, emissive=em) if kind == "box" else _cyl(sz[0], sz[1], rgb))
        for k in range(self.opp_k):
            for nm, kind, sz, _pos, _yaw, rgb, em in _OM:
                self._add(f"op{k}_{nm}", _cuboid(sz, rgb, emissive=em) if kind == "box" else _cyl(sz[0], sz[1], rgb))
        for name, oc in self.objs.items():
            setattr(cfg, name, oc)
        # camera pitched up by cam_tilt about body y (ROS convention: z forward, x right, y down)
        t = self.cam_tilt
        R = np.array([[0.0, math.sin(t), math.cos(t)], [-1.0, 0.0, 0.0], [0.0, -math.cos(t), math.sin(t)]])
        ap = 20.955
        cfg.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Quad/Camera", update_period=0.0, height=self.H, width=self.W,
            update_latest_camera_pose=True,  # Isaac Lab default False: the pose from initialisation is returned, the camera would never follow the vehicle
            data_types=["rgb", "distance_to_image_plane"] if self.depth else ["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=ap / 2 / math.tan(math.radians(self.hfov) / 2),
                                             horizontal_aperture=ap, clipping_range=(0.05, 60.0)),
            offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.03), rot=_mat_to_quat_xyzw(R), convention="ros"),
        )
        self.parts = list(self.objs)

    def bind(self, sim, scene) -> None:
        self.sim, self.scene = sim, scene
        self.cam = scene["camera"]
        self.objs_rt = {k: scene[k] for k in self.parts}
        self.origins = scene.env_origins
        self.all_ids = torch.arange(self.n, device=self.origins.device)

    # ---- per step -----------------------------------------------------------------------------
    def _put(self, name, pos_local, yaw, show, roll=None):
        z = torch.where(show, pos_local[:, 2], torch.full_like(pos_local[:, 2], PARK_Z))
        pos = torch.stack((pos_local[:, 0], pos_local[:, 1], z), -1) + self.origins
        sz, cz = torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)
        if roll is None:
            q = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), sz, cz), -1)  # xyzw
        else:  # yaw about world z, then roll about the gate normal (sign tilted in its own plane)
            sx, cx = torch.sin(0.5 * roll), torch.cos(0.5 * roll)
            q = torch.stack((cz * sx, sz * sx, sz * cx, cz * cx), -1)
        self.objs_rt[name].write_root_pose_to_sim_index(root_pose=torch.cat((pos, q), -1), env_ids=self.all_ids)

    def update(self, env) -> None:
        """Place all gate / marker / pole prims for the current RaceEnv state."""
        dev, n = env.device, env.n
        last = env._last_slot()
        ar = env.arange
        # per-episode, per-gate sign placement jitter (hand-mounted signs are never identical):
        # lateral shift along the bar, tilt in the gate plane, small yaw wobble
        if getattr(self, "jit", None) is None:
            self.jit = torch.zeros(n, env.gate_pos.shape[1], 3, device=dev)
            fresh = torch.ones(n, dtype=torch.bool, device=dev)
        else:
            fresh = env.steps == 0
        if fresh.any():
            r = (torch.rand(int(fresh.sum()), env.gate_pos.shape[1], 3, device=dev) * 2 - 1) * torch.tensor(JITTER, device=dev)
            self.jit[fresh] = r
        half = 0.5 * env.gate_size
        fr = env.cfg.track.gate_frame
        top = half + fr
        zero3 = torch.zeros(n, 3, device=dev)
        for w, rel in enumerate(self.WINDOW):
            idx = torch.minimum((env.gate_idx + rel).clamp_min(0), last)
            valid = ((env.gate_idx + rel) >= 0) & (idx < env.n_gates)
            c = env.gate_pos[ar, idx]
            yaw = env.gate_yaw[ar, idx]
            right = torch.stack((torch.sin(yaw), -torch.cos(yaw), torch.zeros_like(yaw)), -1)
            up = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
            p = f"s{w}"

            def bars(prefix, centre, show):
                off = half + 0.5 * BAR
                for b, (s_, vec) in enumerate(((1, up), (-1, up), (1, right), (-1, right))):
                    self._put(f"{p}_{prefix}{b}", centre + s_ * off[:, None] * vec, yaw, show)

            def corners(prefix, centre, show):  # printed logos on the bottom bar and the two side bars
                lats = (-0.6, 0.6, -(half + 0.5 * BAR), half + 0.5 * BAR)
                for b, lt in enumerate(lats):
                    dz = -(half + 0.5 * BAR) if b < 2 else torch.full_like(half, 0.25)
                    lat = lt if torch.is_tensor(lt) else torch.full_like(half, lt)
                    self._put(f"{p}_{prefix}{b}", centre + lat[:, None] * right + dz[:, None] * up, yaw, show)

            def leds(prefix, centre, show):  # violet LED strip along the inside edge of the opening
                for b, (s_, vec) in enumerate(((1, up), (-1, up), (1, right), (-1, right))):
                    self._put(f"{p}_{prefix}{b}", centre + s_ * (half - 0.025)[:, None] * vec, yaw, show)

            def post(prefix, centre, show):
                self._put(f"{p}_{prefix}", centre - (half + BAR + 1.75)[:, None] * up, yaw, show)

            dshow = valid & env.decoy_on[ar, idx] & env.cfg.markers
            bars("bar", c, valid)
            corners("cor", c, valid)
            leds("led", c, valid)
            post("post", c, valid)
            bars("dbar", env.decoy_pos[ar, idx], dshow)
            corners("dcor", env.decoy_pos[ar, idx], dshow)
            leds("dled", env.decoy_pos[ar, idx], dshow)
            post("dpost", env.decoy_pos[ar, idx], dshow)
            mk = valid & env.cfg.markers
            fin = idx == env.n_gates - 1
            turn = env.m_turn[ar, idx]
            climb = env.m_climb[ar, idx]
            fork = env.m_fork[ar, idx]
            panel = c + (half + 0.5 * BAR)[:, None] * up  # centre of the 0.6 m top bar: marker printed flush on it
            stripe = c - (half + 0.5 * BAR)[:, None] * up  # centre of the bottom bar
            cls = torch.where(fin, torch.full_like(turn, 4), turn)
            jl, jt, jy = self.jit[ar, idx, 0], self.jit[ar, idx, 1], self.jit[ar, idx, 2]
            panel = panel + jl[:, None] * right
            stripe = stripe + (0.8 * jl)[:, None] * right
            yawp = yaw + jy
            ct, st = torch.cos(jt)[:, None], torch.sin(jt)[:, None]
            self._put(f"{p}_pbase", panel, yawp, mk, roll=jt)
            for k_, (dy, dz) in enumerate(PATCH_POS):
                on = torch.zeros(n, dtype=torch.bool, device=dev)
                for cl, ks in PATCH_SET.items():
                    if k_ in ks:
                        on |= cls == cl
                off = (dy * ct - dz * st) * right + (dy * st + dz * ct) * up
                self._put(f"{p}_patch{k_}", panel + off, yawp, mk & on, roll=jt)
            for c_ in range(3):
                self._put(f"{p}_climb{c_}", stripe, yaw, mk & (climb == c_))
            side = torch.where(fork == 1, -1.0, 1.0)[:, None]  # which side the fork sign sits on = which way to go
            fp = c + (half + 0.5 * BAR)[:, None] * up + side * 0.9 * right
            self._put(f"{p}_fork", fp, yaw, mk & (fork != 0))
            self._put(f"{p}_forksq", fp, yaw, mk & (fork != 0))
        for k in range(self.n_poles):
            xy = env.obs_xy[:, k] if k < env.obs_xy.shape[1] else zero3[:, :2]
            on = env.obs_on[:, k] if k < env.obs_on.shape[1] else torch.zeros(n, dtype=torch.bool, device=dev)
            pos = torch.cat((xy, torch.full_like(xy[:, :1], 1.75)), -1)
            self._put(f"pole{k}", pos, torch.zeros(n, device=dev), on)
        land = env.gate_pos[ar, env.n_gates]
        self._put("pad", torch.cat((land[:, :2], torch.full_like(land[:, :1], 0.02)), -1),
                  torch.zeros(n, device=dev), torch.full((n,), bool(env.cfg.landing), device=dev))
        if self.drone_model:
            self._drone(env)
        if self.opp_k and getattr(env, "opp", None) is not None:
            self._opponents(env)
        self.follow(env)

    def _drone(self, env) -> None:
        """Pose the visual drone parts on the vehicle (body pose composed with each part's local offset / yaw)."""
        from droneracing.quad import quat_to_rot
        Rw = quat_to_rot(env.quad.quat)
        for nm, _k, _sz, lp, yaw, _rgb, _em in _DM:
            c, s_ = math.cos(yaw), math.sin(yaw)
            Rl = torch.tensor([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]], device=env.device)
            pos = env.quad.pos + Rw @ torch.tensor(lp, device=env.device) + self.origins
            q = _rot_to_quat_xyzw_torch(Rw @ Rl)
            self.objs_rt[f"dm_{nm}"].write_root_pose_to_sim_index(root_pose=torch.cat((pos, q), -1), env_ids=self.all_ids)

    def _opponents(self, env) -> None:
        """Pose the scripted opponent drones (env.opp) as flat, yaw-only visual bodies; inactive ones are parked below the ground."""
        o = env.opp
        for k in range(self.opp_k):
            show, yaw0 = o.active[:, k], o.yaw[:, k]
            for nm, _kd, _sz, lp, pyaw, _rgb, _em in _OM:
                c, s_ = torch.cos(yaw0), torch.sin(yaw0)
                pos = o.pos[:, k] + torch.stack((c * lp[0] - s_ * lp[1], s_ * lp[0] + c * lp[1], torch.full_like(c, lp[2])), -1)
                self._put(f"op{k}_{nm}", pos, yaw0 + pyaw, show)

    def follow(self, env) -> None:
        """Write the camera world pose from the vehicle state. The camera prim is a child of the rigid body, but with Isaac Lab's
        Fabric frame view its world pose is not refreshed when the parent moves (the camera stays where it was spawned)."""
        cpos, Rc = self.cam_pose(env)
        self.cam.set_world_poses(positions=cpos + self.origins, orientations=_rot_to_quat_xyzw_torch(Rc), convention="ros")

    def render(self, dt: float) -> torch.Tensor:
        """RGB [N,3,H,W] in [0,1] of the current Isaac state."""
        self.sim.render()
        self.cam.update(dt=dt)
        rgb = self.cam.data.output["rgb"][..., :3]
        rgb = rgb.permute(0, 3, 1, 2)
        return rgb.float() / 255.0 if rgb.dtype == torch.uint8 else rgb.float().clamp(0, 1)

    # ---- ground truth for perception training ----------------------------------------------------
    def cam_pose(self, env):
        """Camera centre [N,3] and camera->world rotation [N,3,3] (optical axes x right, y down, z forward), env-local."""
        t = self.cam_tilt
        Rb = torch.tensor([[0.0, math.sin(t), math.cos(t)], [-1.0, 0.0, 0.0], [0.0, -math.cos(t), math.sin(t)]], device=env.device)
        from droneracing.quad import quat_to_rot
        Rw = quat_to_rot(env.quad.quat)
        off = torch.tensor([0.1, 0.0, 0.03], device=env.device)
        Rc = Rw @ Rb
        if self.extr_err is not None:
            Rc = Rc @ self.extr_err
        return env.quad.pos + (Rw @ off), Rc

    def project(self, env, pts):
        """pts [N,K,3] env-local -> pixel (u,v) [N,K,2] and camera-z depth [N,K]."""
        cpos, Rc = self.cam_pose(env)
        pc = (Rc.transpose(1, 2) @ (pts - cpos[:, None]).transpose(1, 2)).transpose(1, 2)
        f = (self.W / 2) / math.tan(math.radians(self.hfov) / 2)
        z = pc[..., 2].clamp_min(1e-3)
        return torch.stack((f * pc[..., 0] / z + self.W / 2, f * pc[..., 1] / z + self.H / 2), -1), pc[..., 2]

    def gate_corners(self, env):
        """Inner corners of the window gates: px [N,4(slot),4(tl,tr,br,bl),2], visible [N,4,4] (in front, in image,
        not hidden behind nearer geometry according to the depth image), slot-valid [N,4]."""
        dev, n = env.device, env.n
        ar, last = env.arange, env._last_slot()
        half = 0.5 * env.gate_size
        px, vis, valid_s = [], [], []
        for rel in self.WINDOW:
            idx = torch.minimum((env.gate_idx + rel).clamp_min(0), last)
            ok = ((env.gate_idx + rel) >= 0) & (idx < env.n_gates)
            c = env.gate_pos[ar, idx]
            yaw = env.gate_yaw[ar, idx]
            right = torch.stack((torch.sin(yaw), -torch.cos(yaw), torch.zeros_like(yaw)), -1)
            up = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
            h = half[:, None]
            pts = torch.stack((c - h * right + h * up, c + h * right + h * up, c + h * right - h * up, c - h * right - h * up), 1)
            uv, z = self.project(env, pts)
            inside = (uv[..., 0] >= 0) & (uv[..., 0] < self.W) & (uv[..., 1] >= 0) & (uv[..., 1] < self.H) & (z > 0.3)
            v = inside & ok[:, None]
            if self.depth and v.any():
                d = self.cam.data.output["distance_to_image_plane"][..., 0]  # [N,H,W]
                ui = uv[..., 0].long().clamp(0, self.W - 1)
                vi = uv[..., 1].long().clamp(0, self.H - 1)
                dd = d[torch.arange(n, device=dev)[:, None], vi, ui]
                v = v & (z <= dd + 0.35)
            px.append(uv); vis.append(v); valid_s.append(ok)
        return torch.stack(px, 1), torch.stack(vis, 1), torch.stack(valid_s, 1)
