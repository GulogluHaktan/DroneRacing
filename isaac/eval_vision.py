"""Full vision stack in Isaac Lab: camera frame -> GateNet -> GRU estimator -> state policy -> CTBR -> PhysX.

The policy never sees privileged gate positions: its gate / velocity / attitude inputs come from the estimator.
Run: scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/t3 --gatenet runs_host/g1/gatenet.pt --estimator runs_host/e0/estimator.pt
"""
import argparse
import json
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, required=True)
parser.add_argument("--gatenet", type=str, required=True)
parser.add_argument("--estimator", type=str, required=True)
parser.add_argument("--n-envs", type=int, default=24)  # 48 envs rendered blank frames on the 8 GB GPU
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--aug", type=float, default=1.0)
parser.add_argument("--oracle-obs", action="store_true", help="policy fed by the env's analytic observation (reference run)")
parser.add_argument("--hfov", type=float, default=130.0)
parser.add_argument("--img-h", type=int, default=192)
parser.add_argument("--img-w", type=int, default=256)
parser.add_argument("--diag", action="store_true", help="measure real GateNet detection rate / PnP error vs ground truth")
parser.add_argument("--crop", type=int, default=0, help="adaptive crop side [px] around the EKF-predicted next gate (needs --ekf3 and a crop-trained GateNet)")
parser.add_argument("--cam-delay", type=int, default=0, help="image latency in policy steps (detections reach the EKF k steps late, uncompensated)")
parser.add_argument("--corrupt", type=float, default=0.0, help="probability that a frame carries an interference band (MonoRace: 8-75 %% of frames)")
parser.add_argument("--rs-ms", type=float, default=0.0, help="rolling-shutter full-frame readout time [ms] (0 = global shutter)")
parser.add_argument("--tilt-err", type=float, default=0.0, help="camera mount error: each Euler angle uniform in +-deg per drone, unknown to the EKF")
parser.add_argument("--no-warmup-abort", action="store_true", help="keep flying when the render warm-up never shows the gates (results are then invalid)")
parser.add_argument("--ekf4", action="store_true", help="EKF3 + corner-count dependent PnP noise (implies --ekf3)")
parser.add_argument("--ablate-gt", action="store_true", help="with --ablate: replace by noise-free ground truth instead of the env oracle obs")
parser.add_argument("--ablate", type=str, default="", help="comma list of vel,up,gate: replace those estimator outputs by the env oracle obs")
parser.add_argument("--dump", type=str, default="", help="closed-loop (policy flies on estimator output) data dump dir for ")
parser.add_argument("--sim-det", type=float, default=None, help="replace GateNet by the analytic detector model with this noise scale (0 = ideal corners, 1 = modelled GateNet)")
parser.add_argument("--ekf", action="store_true", help="map-based EKF (droneracing/ekf.py) instead of the GRU estimator")
parser.add_argument("--map-err", type=float, default=0.25)
parser.add_argument("--map-yaw", type=float, default=0.09)
parser.add_argument("--hall-prob", type=float, default=None, help="override the track mix of the saved run (0 = old open tracks)")
parser.add_argument("--drag-prob", type=float, default=None)
parser.add_argument("--fixed-prob", type=float, default=None, help="fraction of A2RL 2025 track replicas (track.a2rl_2025_lap)")
parser.add_argument("--fixed-jitter", type=float, nargs=2, default=None, help="POS_M YAW_RAD per-gate jitter of the replica")
parser.add_argument("--fixed-scale", type=float, nargs=2, default=None)
parser.add_argument("--fixed-mirror", type=float, default=None)
parser.add_argument("--gateseg", type=str, default="", help="GateSeg checkpoint: frame segmentation + edge-line corners (MonoRace style) replaces GateNet detections")
parser.add_argument("--video", type=str, default="", help="record per-episode clips (chase view + onboard view with true/estimated next gate) into this dir")
parser.add_argument("--drone-model", action="store_true", help="video only: render a visual racing-drone body (arms, motors, props, canopy) instead of the tiny cube")
parser.add_argument("--video-envs", type=int, default=8, help="record the first N drones")
parser.add_argument("--video-max", type=int, default=2, help="clips kept per outcome (finished / crashed)")
parser.add_argument("--aug-stable", action="store_true", help="draw the appearance augmentation (sky, floor tint, gain, gamma) once per drone and keep it for the whole run: a real camera with fixed exposure has no frame-to-frame colour flicker")
parser.add_argument("--ekf-delay", type=int, default=-1, help="delay (policy steps) assumed by the EKF5 compensation; -1 = same as --cam-delay (set different to test a delay mismatch)")
parser.add_argument("--mr", type=str, default="", help="MonoRace perception: GateNetMR checkpoint (adaptive crop, mask, QuAdGate corners with identity) + GateEKF8; implies --ekf5")
parser.add_argument("--tilt-seed", type=int, default=-1, help="seed of the per-drone mount error (--tilt-err); -1 = track seed + 1000 (old behaviour). Fix it to fly the same drones (same mounts) on other tracks")
parser.add_argument("--extr-corr", type=str, default="", help="per-drone mount rotation estimate [n_envs,3] deg (scripts/calib_extrinsic.py --out); detections are corrected before the filter (droneracing/extrinsic.py)")
parser.add_argument("--exposure-aug", type=float, default=0.0, help="fraction of frames with a short exposure (dark, gain noise, 10-bit) + lens-edge blur (droneracing/realcam_aug.exposure_edge)")
parser.add_argument("--percep-log", type=str, default="", help="per camera frame: next-gate size, corners in image, view angle, body rate, distance, whole-gate detection found / corner error (to calibrate droneracing/detector_model.py)")
parser.add_argument("--save-crops", type=str, default="", help="on-policy detector data: store the clean render crop of every camera frame (crop line) with true corner labels, collect_percep format")
parser.add_argument("--crop-gt", action="store_true", help="diagnostic: crop window from the true next-gate corners instead of the filter prediction")
parser.add_argument("--derotate-gt", action="store_true", help="diagnostic: derotate with the true camera roll instead of the filter roll")
parser.add_argument("--derotate", action="store_true", help="GateNet input rotated by -roll of the filter attitude (net trained with train_gatenet2 --derotate); detections rotated back")
parser.add_argument("--peak-thr", type=float, default=0.3, help="score threshold of the GateNet corner peaks (decode_corner_peaks)")
parser.add_argument("--rs-comp", type=float, default=0.0, help="rolling-shutter compensation of the corner measurements with the gyro: readout time [ms] assumed by the filter (droneracing/rscomp.py)")
parser.add_argument("--min-det-dist", type=float, default=0.0, help="drop all detections while the filter-estimated distance to the next gate is below this [m]: the filter coasts on the IMU through the last metres (KAIST drops detections < 1 m)")
parser.add_argument("--band-check", action="store_true", help="drop only the detections inside corrupt row bands (droneracing/framecheck.corrupt_rows) instead of the whole frame")
parser.add_argument("--frame-check", action="store_true", help="reject corrupt frames (noise / smeared band, droneracing/framecheck.py): the detections of the frame are dropped")
parser.add_argument("--mr-crop", type=int, default=192)
parser.add_argument("--mr-gates", type=int, default=2)
parser.add_argument("--ekf6", action="store_true", help="GateEKF6: GateEKF5 + online camera mount rotation estimate (implies --ekf5)")
parser.add_argument("--ekf5", action="store_true", help="GateEKF5: GateEKF3 with image-delay compensation (delay = --cam-delay policy steps); implies --ekf3")
parser.add_argument("--cam-hz-exact", action="store_true", help="fractional camera rate: a frame on every policy step where floor(t * cam_hz) changes (60 Hz at a 100 Hz policy = 3 frames in 5 steps); default keeps the old integer period")
parser.add_argument("--cam-hz", type=float, default=0.0, help="camera frame rate [Hz]: the filter gets a detection only every round(policy_hz/cam_hz) policy steps (own camera: 30)")
parser.add_argument("--yolo", type=str, default="", help="YOLOv8-pose gate keypoint detector (KAIST style, droneracing/yolopose.py) replaces GateNet detections")
parser.add_argument("--yolo-imgsz", type=int, default=256)
parser.add_argument("--decoy-prev", action="store_true", help="EKF3: drop detections of the gate just passed (GateEKF3.mask_decoys)")
parser.add_argument("--save-frames", type=str, default="", help="save a montage of rendered frames (steps 5, 150, 400) to this dir")
parser.add_argument("--cam-explicit", action="store_true", help="write the camera world pose every step from the vehicle state (instead of relying on the camera prim following the rigid body)")
parser.add_argument("--rtx-fabric-off", action="store_true", help="render delegate reads USD transforms instead of Fabric (dynamic objects / camera follow)")
parser.add_argument("--geo-budget", type=int, default=0, help="geometry streaming GPU budget [MB] (0 = Kit default 64000)")
parser.add_argument("--ekf3", action="store_true", help="GateEKF3: also use single corner peaks of the (refine) GateNet for gates whose centre is not detected")
parser.add_argument("--ekf-args", type=str, default="{}")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--opp-head", type=str, default="", help="opponent head checkpoint (droneracing/oppnet.py, on the GateNet decoder feature): the policy's opponent block comes from camera detections + a world-frame tracker instead of the simulator (needs --crop and --ekf)")
parser.add_argument("--opp-thr", type=float, default=0.4)
parser.add_argument("--opp-wide", action="store_true", help="second opponent pass: whole frame letterboxed into the crop window (GateNet trunk + opponent head only); detections merged with the crop pass")
parser.add_argument("--opp-hits", type=int, default=1, help="a track reaches the policy only after this many detections")
parser.add_argument("--opp-prob", type=float, default=None, help="override the fraction of episodes with scripted opponents (0 = none)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app
if args_cli.rtx_fabric_off or args_cli.geo_budget:
    import carb
    _st = carb.settings.get_settings()
    if args_cli.rtx_fabric_off:
        _st.set("/rtx/hydra/readTransformsFromFabricInRenderDelegate", False)
    if args_cli.geo_budget:
        _st.set("/rtx/hydra/geometrystreaming/gpuBudgetMB", int(args_cli.geo_budget))

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, "/workspace/DroneRacing")
sys.path.insert(0, "/workspace/DroneRacing/isaac")
sys.path.insert(0, "/workspace/DroneRacing/scripts")
from droneracing import RaceEnv  # noqa: E402
from droneracing.config import make_cfg  # noqa: E402
from droneracing.detector_model import project_corners  # noqa: E402
from droneracing.estimator import Estimator, Strapdown, det_features, split_out  # noqa: E402
from droneracing.gatenet import GateNet, decode  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402
from droneracing.ppo import ActorCritic, PPOCfg  # noqa: E402
from eval_state import summarize  # noqa: E402
from isaac_quad import IsaacQuad, build_scene  # noqa: E402
from vision_rig import VisionRig  # noqa: E402

a = args_cli
torch.manual_seed(a.seed)
root = Path("/workspace/DroneRacing")
pdir = root / a.policy
meta = json.loads((pdir / "config.json").read_text())
cfg = make_cfg(dict(meta["args"]), a.n_envs, policy_sees_params=meta["env"]["policy_sees_params"])
cfg.mid_init_prob = 0.0
cfg.resid_bank = ""  # the camera + estimator replace the simulated errors; the bank path is host-relative anyway
if args_cli.opp_prob is not None:
    cfg.opp_prob = args_cli.opp_prob
if args_cli.hall_prob is not None:
    cfg.track.hall_prob = args_cli.hall_prob
if args_cli.drag_prob is not None:
    cfg.track.drag_prob = args_cli.drag_prob
if args_cli.fixed_prob is not None:
    cfg.track.fixed_prob = args_cli.fixed_prob
    cfg.track.max_gates = max(cfg.track.max_gates, 22)
    if args_cli.fixed_jitter is not None:
        cfg.track.fixed_jitter = tuple(args_cli.fixed_jitter)
    if args_cli.fixed_scale is not None:
        cfg.track.fixed_scale = tuple(args_cli.fixed_scale)
    if args_cli.fixed_mirror is not None:
        cfg.track.fixed_mirror = args_cli.fixed_mirror
if args_cli.ekf:
    cfg.sensors.gate_bias, cfg.sensors.gate_yaw_bias = args_cli.map_err, args_cli.map_yaw
rig = VisionRig(a.n_envs, a.img_h, a.img_w, hfov_deg=a.hfov, cam_tilt=cfg.cam_tilt)
rig.drone_model = a.drone_model
if a.opp_head:
    from droneracing.oppnet import OppHead, decode as opp_decode
    _oh = OppHead().to("cuda")
    _oh.load_state_dict(torch.load(root / a.opp_head, map_location="cuda")["head"])
    _oh.eval()
    opp_tracks = [[] for _ in range(a.n_envs)]
    opp_stats = {"frames": 0, "dets": 0, "tp": 0, "fp": 0, "vis": 0, "found": 0}
rig.opp_k = cfg.opp_max  # scripted opponents are rendered (orange drones); the policy gets their noisy simulated observation block
CW_, CH_ = 384, 216


def _attach_all(c):
    rig.attach(c)
    if a.video:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import TiledCameraCfg
        c.chase = TiledCameraCfg(prim_path="{ENV_REGEX_NS}/ChaseCam", update_period=0.0, height=CH_, width=CW_, data_types=["rgb"],
                                 spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.05, 200.0)))


sim, scene = build_scene(cfg.n_envs, cfg.dt, "cuda:0", env_spacing=150.0, extra_cfg=_attach_all,
                         ground_size=150.0 * (int(a.n_envs ** 0.5) + 2))
rig.bind(sim, scene)
vid = {"eye": None, "buf": {}, "cnt": {"finish": 0, "crash": 0}}
if a.video:
    import io as _io

    import torch.nn.functional as _F
    from PIL import Image as _Im, ImageDraw as _Dr
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
_ts_g, _tc_g = math.sin(cfg.cam_tilt), math.cos(cfg.cam_tilt)
if a.gateseg:
    from droneracing.gateseg import GateSeg
    gs = GateSeg().to(dev)
    gs.load_state_dict(torch.load(root / a.gateseg, map_location=dev)["net"])
    gs.eval()
if a.mr:
    from droneracing.monorace import GateNetMR
    _mck = torch.load(root / a.mr, map_location=dev)
    mrnet = GateNetMR(_mck.get("f", 4)).to(dev)
    mrnet.load_state_dict(_mck["net"])
    mrnet.eval()
if a.yolo:
    from ultralytics import YOLO
    yolo = YOLO(str(root / a.yolo))
ck = torch.load(root / a.estimator, map_location=dev)
est = Estimator(ck.get("hidden", 160)).to(dev)
est.load_state_dict(ck["net"])
est.eval()
assert cfg.vis_obs and cfg.n_obs_obstacles >= 0 and not cfg.markers and not cfg.landing
assert cfg.track.max_obstacles == 0, "vision policy expects tracks without poles"
K = cfg.obs_history
if a.ekf:
    from droneracing.ekf import GateEKF
    from droneracing.quad import quat_to_rot as quat_to_rot_
    from droneracing.quad import quat_exp, quat_mul
    if a.mr:
        a.ekf5 = a.ekf3 = True
        from droneracing.ekf8 import GateEKF8 as GateEKF  # noqa: N814
    elif a.ekf6:
        a.ekf5 = a.ekf3 = True
        from droneracing.ekf6 import GateEKF6 as GateEKF  # noqa: N814
    elif a.ekf5:
        a.ekf3 = True
        from droneracing.ekf5 import GateEKF5 as GateEKF  # noqa: N814
    elif a.ekf4:
        a.ekf3 = True
        from droneracing.ekf4 import GateEKF4 as GateEKF  # noqa: N814
    elif a.ekf3:
        from droneracing.ekf3 import GateEKF3 as GateEKF  # noqa: N814
    ekf = GateEKF(a.n_envs, dev, sig_map=max(a.map_err, 0.05), H=a.img_h, W=a.img_w, hfov=a.hfov, cam_tilt=cfg.cam_tilt, **json.loads(a.ekf_args))

    def ekf_reset(ids):
        k = len(ids)
        q = quat_mul(env.quad.quat[ids], quat_exp(torch.randn(k, 3, device=dev) * 0.02))
        ekf.reset(ids, env.quad.pos[ids] + torch.randn(k, 3, device=dev) * 0.05, env.quad.vel[ids], q)

    def map_gates(J=3):
        last = env._last_slot()
        idx = [torch.minimum(env.gate_idx + j, last) for j in range(J)]
        cs, ns = zip(*[env._gate_frame(i, noisy=True)[:2] for i in idx])
        return torch.stack(cs, 1), torch.stack(ns, 1), last

    ekf_reset(torch.arange(a.n_envs, device=dev))
n = a.n_envs
dt = cfg.dt * cfg.decimation


def _rerot(dets: list, pks: list, roll: torch.Tensor, Hh: int, Ww: int):
    """--derotate: detections of the derotated image (dets rows: score + 4 corners px; peaks rows: type, score, u, v) -> original image."""
    from droneracing.derotate import rerotate_points
    od, op = [], []
    for b, (d_, p_) in enumerate(zip(dets, pks)):
        if len(d_):
            d_ = d_.clone()
            d_[:, 1:] = rerotate_points(d_[:, 1:].reshape(1, -1, 2), roll[b:b + 1], Hh, Ww).reshape(-1, 8)
        if len(p_):
            p_ = p_.clone()
            p_[:, 2:4] = rerotate_points(p_[None, :, 2:4], roll[b:b + 1], Hh, Ww)[0]
        od.append(d_); op.append(p_)
    return od, op


def _frame_now(step: int) -> bool:
    """--cam-hz-exact: True on the policy steps where a new camera frame starts (fractional rate, e.g. 60 Hz at a 100 Hz policy)."""
    return step == 0 or math.floor(step * dt * a.cam_hz + 1e-9) != math.floor((step - 1) * dt * a.cam_hz + 1e-9)


H, W = a.img_h, a.img_w
fx = (W / 2) / math.tan(math.radians(a.hfov) / 2)
if a.video:
    _vs, _vc = math.sin(cfg.cam_tilt), math.cos(cfg.cam_tilt)
    _RB = torch.tensor([[0.0, _vs, _vc], [-1.0, 0.0, 0.0], [0.0, -_vc, _vs]], device=dev)  # camera -> body axes
    _fxc = 14.0 / 20.955 * CW_
    _vdir = root / a.video
    _vdir.mkdir(parents=True, exist_ok=True)
    _VN = min(a.video_envs, a.n_envs)
    vid["buf"] = {k: [] for k in range(_VN)}
    vid["trail"] = {k: [] for k in range(_VN)}

    def chase_pose():
        pos, vel = env.quad.pos, env.quad.vel
        sp = vel[:, :2].norm(dim=-1, keepdim=True)
        tdir = torch.where(sp > 1.0, vel[:, :2] / sp.clamp_min(1e-3), torch.tensor([[1.0, 0.0]], device=dev).expand(n, 2))
        cur = tdir if vid["eye"] is None else _F.normalize(0.9 * vid["eye"] + 0.1 * tdir, dim=-1)
        vid["eye"] = cur
        eye = pos + torch.cat((-(1.7 if a.drone_model else 3.4) * cur, torch.full_like(cur[:, :1], 0.55 if a.drone_model else 1.4)), -1)
        eye[:, 2].clamp_(min=0.6)
        tgt = pos + torch.cat((1.5 * cur, torch.zeros_like(cur[:, :1])), -1)
        chase.set_world_poses_from_view(eye + scene.env_origins, tgt + scene.env_origins)
        vid["eyeT"] = (eye, tgt)

    def _gate_poly(rel_b, nrm_b, up_b, half=0.75):
        right = _F.normalize(torch.linalg.cross(nrm_b, up_b, dim=-1), dim=-1)
        pts = torch.stack((rel_b - half * right + half * up_b, rel_b + half * right + half * up_b,
                           rel_b + half * right - half * up_b, rel_b - half * right - half * up_b), 0)
        pc = (pts - torch.tensor((0.1, 0.0, 0.03), device=dev)) @ _RB
        if bool((pc[:, 2] < 0.3).any()):
            return None
        return [((fx * p[0] / p[2] + W / 2).item(), (fx * p[1] / p[2] + H / 2).item()) for p in pc]

    def capture(img_, rel_, nrm_, up_, R_, step_):
        eye, tgt = vid["eyeT"]
        idx = torch.minimum(env.gate_idx, env._last_slot())
        c_t, n_t, _ = env._gate_frame(idx)
        rt = (R_.transpose(1, 2) @ (c_t - env.quad.pos).unsqueeze(-1)).squeeze(-1)
        nt = (R_.transpose(1, 2) @ n_t.unsqueeze(-1)).squeeze(-1)
        upt = R_[:, 2, :]
        ch = chase.data.output["rgb"]
        for k in range(_VN):
            c = _Im.fromarray(ch[k, ..., :3].clamp(0, 255).byte().cpu().numpy())
            f_ = _F.normalize(tgt[k] - eye[k], dim=0)
            r_ = _F.normalize(torch.linalg.cross(f_, torch.tensor([0.0, 0.0, 1.0], device=dev)), dim=0)
            u_ = torch.linalg.cross(r_, f_)

            def proj(p):
                dd = p - eye[k]
                z = float(dd @ f_)
                return (CW_ / 2 + _fxc * float(dd @ r_) / z, CH_ / 2 - _fxc * float(dd @ u_) / z) if z > 0.2 else None

            vid["trail"][k].append(env.quad.pos[k].clone())
            del vid["trail"][k][:-40]
            d1 = _Dr.Draw(c)
            pts = [q for q in (proj(t) for t in vid["trail"][k]) if q is not None]
            if len(pts) > 1:
                d1.line(pts, fill=(255, 160, 0), width=2)
            if pts:
                x, y = pts[-1]
                if not a.drone_model:
                    d1.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(255, 60, 0), width=3)
            fi = _Im.fromarray((img_[k].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()).resize((384, 288), _Im.NEAREST)
            d2 = _Dr.Draw(fi)
            sx, sy = 384 / W, 288 / H
            pt = _gate_poly(rt[k], nt[k], upt[k])
            if pt is not None:
                d2.polygon([(x * sx, y * sy) for x, y in pt], outline=(0, 255, 0))
            pe = _gate_poly(rel_[k, 0] * 5.0, nrm_[k, 0], up_[k])
            if pe is not None:
                d2.polygon([(x * sx, y * sy) for x, y in pe], outline=(0, 255, 255))
            ge = float((rel_[k, 0] * 5.0 - rt[k]).norm())
            row = _Im.new("RGB", (CW_ + 384, 288 + 20), (0, 0, 0))
            row.paste(c, (0, 0)); row.paste(fi, (CW_, 0))
            dr = _Dr.Draw(row)
            dr.text((4, 290), f"step {step_}  gate {int(env.gate_idx[k])}/{int(env.n_gates[k])}  v {float(env.quad.vel[k].norm()):4.1f} m/s  dist {float(rt[k].norm()):4.1f} m  "
                              f"belief error {ge:4.2f} m   GREEN = true gate, CYAN = gate the policy believes", fill=(255, 255, 255))
            bio = _io.BytesIO()
            row.save(bio, format="JPEG", quality=85)
            vid["buf"][k].append(bio.getvalue())

    def end_episode(k, fin, cr, gates_, ng_):
        kind = "finish" if fin else ("crash" if cr else "timeout")
        if kind in vid["cnt"] and vid["cnt"][kind] < a.video_max and len(vid["buf"][k]) > 20:
            d = _vdir / f"{kind}_env{k}_{vid['cnt'][kind]}"
            d.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(vid["buf"][k]):
                (d / f"{i:05d}.jpg").write_bytes(b)
            (d / "meta.txt").write_text(f"{kind} env {k} gates {gates_}/{ng_} frames {len(vid['buf'][k])}\n")
            vid["cnt"][kind] += 1
            print(f"VIDEO clip {kind} env {k} gates {gates_}/{ng_} frames {len(vid['buf'][k])} -> {d}", flush=True)
        vid["buf"][k] = []
        vid["trail"][k].clear()

def set_cam_pose():
    """Camera world pose from the torch-side vehicle state (same camera model as VisionRig.cam_pose), written explicitly."""
    from vision_rig import _mat_to_quat_xyzw  # noqa: F401
    cpos, Rc = rig.cam_pose(env)
    q = torch.stack([torch.tensor(_mat_to_quat_xyzw(Rc[i].cpu().numpy()), device=dev, dtype=torch.float32) for i in range(env.n)])
    rig.cam.set_world_poses(positions=cpos + rig.origins, orientations=q, convention="ros")

if a.tilt_err > 0:
    from cam_artifacts import random_rotation
    _g = torch.Generator(device=dev); _g.manual_seed(a.seed + 1000 if a.tilt_seed < 0 else a.tilt_seed)
    rig.extr_err = random_rotation(a.n_envs, a.tilt_err, dev, _g)
import collections as _co
dq = _co.deque(maxlen=a.cam_delay + 1)
rig.update(env)
if not a.oracle_obs and a.sim_det is None:  # the renderer shows gates only after its geometry streaming caught up (slow while the GPU is busy): wait until the gates really appear
    import time as _t
    _t0 = _t.time()
    _ok = False
    while _t.time() - _t0 < 300:
        _f = rig.render(dt)
        _px, _vis, _val = rig.gate_corners(env)
        _m = (_vis[:, 1].all(-1) & _val[:, 1])  # next gate fully inside the image according to the geometry
        if int(_m.sum()) < 3:
            _ok = True
            break
        _std = _f.float().std(dim=(1, 2, 3))
        if float((_std[_m] > 0.03).float().mean()) >= 0.7:
            _ok = True
            break
        _t.sleep(2.0)
    print(f"render warm-up: {_t.time() - _t0:.0f} s, gates rendered: {_ok}", flush=True)
    if not _ok and not a.no_warmup_abort:  # blank frames give a bogus ~0 % result (seen with a trainer on the same GPU)
        print("Error: WARMUP FAILED, gates never rendered; is another job using the GPU?", flush=True)
        app.close()
        sys.exit(3)
_causes = {}  # crash cause -> count (1 frame, 2 pole, 4 ground, 6 opponent, see RaceEnv.step)
obs, _ = env.compute_obs()
h = None
strap = Strapdown(a.n_envs, dev)
hist = torch.zeros(n, K, 10, device=dev)
age_t = torch.ones(n, 64, device=dev)  # time since the EKF last accepted a measurement of each gate / 3 s (age_obs policies)
dg_prev = None
seen = torch.zeros(n, dtype=torch.bool, device=dev)
rec = {k: torch.zeros(n, device=dev) for k in ("fin", "crash", "len", "twr", "tlen", "gates", "ng", "jitter")}
rec["mspeed"] = torch.zeros(n, device=dev)  # mean TRUE speed of the first episode of each drone [m/s] (path-speed uses the reference track length)
speed_sum = torch.zeros(n, device=dev)
vel_err, up_err, g_err, g_n, steps_alive = 0.0, 0.0, 0.0, 0.0, 0.0
rh_blank, rh_frozen, rh_n, rh_prev = 0, 0, 0, None
gerr_bins = [[], [], [], []]
prev_act = None
buf = {k: [] for k in ("feat", "imu", "act", "vel", "up", "gate", "done", "valid", "est", "dbg")}
since_reset = torch.zeros(n, dtype=torch.long, device=dev)
diag = {"vis": [], "found": [], "err": [], "dist": [], "nerr": [], "cerr": [], "ext": [], "nin": []}
with torch.no_grad():
    for step in range(a.steps):
        if a.oracle_obs:
            pol_obs = obs
        else:
            if a.cam_explicit:
                set_cam_pose()
            if a.video:
                chase_pose()
            rgb = None if a.sim_det is not None else rig.render(dt)
            if a.video:
                chase.update(dt=dt)
            if rgb is not None:  # render health: blank frames, and frames identical to the previous step (renderer stalled)
                _fs = rgb.float().std(dim=(1, 2, 3))
                _sc = 255.0 if float(rgb.max()) > 1.5 else 1.0
                rh_blank += int((_fs / _sc < 0.01).sum())
                if rh_prev is not None:
                    rh_frozen += int(((rgb.float() - rh_prev).abs().mean(dim=(1, 2, 3)) / _sc < 1e-5).sum())
                rh_prev, rh_n = rgb.float().clone(), rh_n + n
            om, v = env.quad.omega, env.quad.vel
            expo = 0.002 + 0.004 * torch.rand(n, device=dev)
            sgn = torch.where(torch.rand(n, 2, device=dev) < 0.5, -1.0, 1.0)
            flow = torch.stack((fx * expo * (om[:, 2].abs() + v.norm(dim=-1) / 6.0), fx * expo * om[:, 1].abs()), -1) * sgn
            if a.sim_det is not None:
                from droneracing.detector_model import sim_detections
                feat = sim_detections(env, None, H, W, noise_scale=a.sim_det)
            else:
                if a.aug_stable:
                    _ag = torch.Generator(device=dev)
                    _ag.manual_seed(4242 + a.seed)
                    img = realistic_augment(rgb, a.aug, flow_px=flow, gen=_ag)
                else:
                    img = realistic_augment(rgb, a.aug, flow_px=flow)
                if a.exposure_aug > 0:
                    from droneracing.realcam_aug import exposure_edge
                    img = exposure_edge(img, None, dark_prob=a.exposure_aug, edge_prob=0.5)
                if a.rs_ms > 0:
                    from cam_artifacts import rolling_shutter
                    _ts, _tc = math.sin(cfg.cam_tilt), math.cos(cfg.cam_tilt)
                    _Rb = torch.tensor([[0.0, _ts, _tc], [-1.0, 0.0, 0.0], [0.0, -_tc, _ts]], device=dev)
                    img = rolling_shutter(img, env.quad.omega @ _Rb, fx, a.rs_ms / 1000.0)
                if a.corrupt > 0:
                    from cam_artifacts import corrupt
                    img = corrupt(img, a.corrupt)
                if a.mr:  # MonoRace: adaptive crop -> GateNetMR mask -> QuAdGate corners matched to the priors of the filter -> GateEKF8
                    from droneracing.adaptive_crop import crop_windows, extract
                    from droneracing.monorace import quadgate, predict_masks
                    import numpy as _np
                    peaks = torch.zeros(n, 16, 5, device=dev)
                    feat = det_features([torch.zeros(0, 9)] * n, H, W, dev, cam_tilt=cfg.cam_tilt, hfov_deg=a.hfov)
                    _per = max(1, round(1.0 / (cfg.dt * cfg.decimation) / a.cam_hz)) if a.cam_hz > 0 else 1
                    if (_frame_now(step) if a.cam_hz_exact else step % _per == 0):
                        _mc, _mn, _last = map_gates()
                        _dly = 0  # the frame is captured now: priors from the current state; the delay is applied to the measurement afterwards (EKF5 roll-back)
                        _uv8, _ok8, _roll = ekf.priors8(env.gate_idx, _mc, _mn, J=a.mr_gates, delay=_dly)
                        # window around gate 0 (and gate 1 when it is visible and close)
                        _okw = _ok8[:, 0].all(-1)
                        _u1 = torch.cat((_uv8[:, 0], _uv8[:, 0]), 1)
                        _pts = _u1
                        if a.mr_gates > 1:
                            _u2 = torch.cat((_uv8[:, 0], _uv8[:, 1]), 1)
                            _w2 = (_u2.amax(1) - _u2.amin(1)).amax(-1)
                            _both = _ok8[:, 1].all(-1) & _okw & (_w2 < 1.8 * a.mr_crop)
                            _pts = torch.where(_both[:, None, None], _u2, _u1)
                        _inb = ((_pts[..., 0].amax(1) > 0) & (_pts[..., 0].amin(1) < W) & (_pts[..., 1].amax(1) > 0) & (_pts[..., 1].amin(1) < H))
                        _win = crop_windows(_pts, _okw & _inb, H, W, a.mr_crop, margin=1.3)
                        _prob = predict_masks(mrnet, extract(img, _win, a.mr_crop))
                        _sc = (a.mr_crop / _win[:, 2]).cpu().numpy()
                        _wn = _win.cpu().numpy()
                        _uvn, _okn, _rn = _uv8.cpu().numpy(), _ok8.cpu().numpy(), _roll.cpu().numpy()
                        for _b in range(n):
                            _pr, _g, _t = [], [], []
                            for _j in range(a.mr_gates):
                                for _k in range(8):
                                    if _okn[_b, _j, _k]:
                                        _x = (_uvn[_b, _j, _k, 0] - _wn[_b, 0]) * _sc[_b]
                                        _y = (_uvn[_b, _j, _k, 1] - _wn[_b, 1]) * _sc[_b]
                                        if -0.1 * a.mr_crop <= _x < 1.1 * a.mr_crop and -0.1 * a.mr_crop <= _y < 1.1 * a.mr_crop:
                                            _pr.append((_x, _y)); _g.append(_j); _t.append(_k)
                            if len(_pr) >= 3:
                                _o = quadgate(_prob[_b], _np.array(_pr), _np.array(_g), _np.array(_t), float(_rn[_b]))
                                for _m, _r in enumerate(_o[:16]):
                                    peaks[_b, _m, 0], peaks[_b, _m, 1] = float(env.gate_idx[_b]) + float(_r[0]), float(_r[1])  # absolute gate slot (the gate index may advance during the delay)
                                    peaks[_b, _m, 2] = float(_wn[_b, 0] + _r[2] / _sc[_b])
                                    peaks[_b, _m, 3] = float(_wn[_b, 1] + _r[3] / _sc[_b])
                                    peaks[_b, _m, 4] = 1.0
                elif a.yolo:  # KAIST style: YOLOv8-pose keypoints, no extrapolation of corners outside the image
                    from droneracing.yolopose import detect as yolo_detect
                    feat = det_features([d_.to(dev) for d_ in yolo_detect(yolo, img, a.yolo_imgsz)], H, W, dev, cam_tilt=cfg.cam_tilt, hfov_deg=a.hfov)
                    peaks = torch.zeros(n, 16, 4, device=dev)
                elif a.gateseg:  # frame mask + edge lines -> inner corners (extrapolated for partial gates); no single-corner peaks
                    from droneracing.gateseg import detect as gs_detect
                    _roll = None
                    if a.ekf:  # derotate with the filter attitude to get the gate-frame corner order (MonoRace)
                        _upc = ekf.rot()[:, 2, :] @ torch.tensor([[0.0, _ts_g, _tc_g], [-1.0, 0.0, 0.0], [0.0, -_tc_g, _ts_g]], device=dev)
                        _roll = torch.atan2(_upc[:, 0], -_upc[:, 1])
                    feat = det_features([d_.to(dev) for d_ in gs_detect(gs, img, roll=_roll)], H, W, dev, cam_tilt=cfg.cam_tilt, hfov_deg=a.hfov)
                    peaks = torch.zeros(n, 16, 4, device=dev)
                elif a.ekf3:
                    from droneracing.gatenet import decode_refined
                    from droneracing.gatenet2 import decode_corner_peaks
                    if a.crop > 0:  # adaptive window around the EKF-predicted next gate (MonoRace); detections mapped back to full-frame px
                        from droneracing.adaptive_crop import crop_windows, dets_to_full, extract, peaks_to_full
                        _mc, _mn, _last = map_gates()
                        _cw = ekf._corners_world(_mc + ekf.dg_for(env.gate_idx, _last, _mc.shape[1]), _mn)
                        _uv, _, _Xc = ekf._project(_cw, ekf.rot())
                        _uv0, _z0 = _uv[:, 0], _Xc[:, 0, :, 2]
                        _okc = (_z0 > 0.3).all(-1) & (_uv0[..., 0].amax(-1) > 0) & (_uv0[..., 0].amin(-1) < W) & (_uv0[..., 1].amax(-1) > 0) & (_uv0[..., 1].amin(-1) < H)
                        if a.crop_gt:  # diagnostic: true next-gate corners (true camera)
                            _gpx_c, _gv_c, _gsv_c = rig.gate_corners(env)
                            _uv0 = _gpx_c[:, 1]
                            _okc = _gsv_c[:, 1] & (_uv0[..., 0].amax(-1) > 0) & (_uv0[..., 0].amin(-1) < W) & (_uv0[..., 1].amax(-1) > 0) & (_uv0[..., 1].amin(-1) < H)
                        _win = crop_windows(_uv0, _okc, H, W, a.crop)
                        _in = extract(img, _win, a.crop)
                        if a.save_crops and ((_frame_now(step) if a.cam_hz_exact else step % max(1, round(1.0 / (cfg.dt * cfg.decimation) / a.cam_hz)) == 0) if a.cam_hz > 0 else True):
                            from droneracing.adaptive_crop import to_crop as _to_crop
                            _gpx_s, _gv_s, _gsv_s = rig.gate_corners(env)  # true camera, slots of the window gates
                            _pc = _to_crop(_gpx_s.reshape(n, -1, 2), _win, a.crop).reshape(_gpx_s.shape)
                            _ins = (_pc[..., 0] >= 0) & (_pc[..., 0] < a.crop) & (_pc[..., 1] >= 0) & (_pc[..., 1] < a.crop)
                            _sc_buf = globals().setdefault("_saved_crops", {k: [] for k in ("img", "corners", "vis", "slot_valid", "state", "win")})
                            _sc_buf["img"].append((extract(rgb, _win, a.crop).clamp(0, 1) * 255).to(torch.uint8).cpu())
                            _sc_buf["corners"].append((_pc / a.crop).half().cpu())
                            _sc_buf["vis"].append((_gv_s & _ins).cpu())
                            _sc_buf["slot_valid"].append(_gsv_s.cpu())
                            _sc_buf["state"].append(torch.cat((env.quad.pos, env.quad.vel, env.quad.quat, env.quad.omega), -1).cpu())
                            _sc_buf["win"].append(_win.cpu())
                        if a.derotate:
                            from droneracing.derotate import derotate_img, roll_from_rot
                            _roll = roll_from_rot(quat_to_rot_(env.quad.quat) if a.derotate_gt else ekf.rot(), cfg.cam_tilt)
                            _in = derotate_img(_in, _roll)
                        _o = gn(_in)
                        _dets, _pk = decode_refined(*_o), decode_corner_peaks(_o[2], _o[3], thr=a.peak_thr)
                        if a.derotate:
                            _dets, _pk = _rerot(_dets, _pk, _roll, a.crop, a.crop)
                        if a.opp_head:
                            from droneracing.derotate import rerotate_points as _rrp
                            _opp_dets = []
                            for _b, _d in enumerate(opp_decode(_oh(gn._y), a.opp_thr)):
                                if len(_d):
                                    _uvd = _rrp(_d[:, 1:3][None], _roll[_b:_b + 1], a.crop, a.crop)[0] if a.derotate else _d[:, 1:3]
                                    _uvd = _win[_b, :2] + _uvd * (_win[_b, 2] / a.crop)
                                    _d = torch.cat((_uvd, (_d[:, 3] * _win[_b, 2] / a.crop)[:, None]), -1)
                                else:
                                    _d = torch.zeros(0, 3, device=dev)
                                _opp_dets.append(_d)
                            if a.opp_wide:  # whole frame letterboxed into the same window size: finds opponents outside the gate crop
                                _win2 = crop_windows(_uv0, torch.zeros_like(_okc), H, W, a.crop)
                                _in2 = extract(img, _win2, a.crop)
                                if a.derotate:
                                    _in2 = derotate_img(_in2, _roll)
                                gn(_in2)
                                for _b, _d in enumerate(opp_decode(_oh(gn._y), a.opp_thr)):
                                    if len(_d):
                                        _uvd = _rrp(_d[:, 1:3][None], _roll[_b:_b + 1], a.crop, a.crop)[0] if a.derotate else _d[:, 1:3]
                                        _uvd = _win2[_b, :2] + _uvd * (_win2[_b, 2] / a.crop)
                                        _d = torch.cat((_uvd, (_d[:, 3] * _win2[_b, 2] / a.crop)[:, None]), -1)
                                        if len(_opp_dets[_b]):  # drop wide detections that repeat a crop detection
                                            _dup = (_d[:, None, :2] - _opp_dets[_b][None, :, :2]).norm(dim=-1).amin(-1) < 25.0
                                            _d = _d[~_dup]
                                        _opp_dets[_b] = torch.cat((_opp_dets[_b], _d), 0)
                        feat = det_features(dets_to_full(_dets, _win, a.crop), H, W, dev, cam_tilt=cfg.cam_tilt, hfov_deg=a.hfov)
                        _pk = peaks_to_full(_pk, _win, a.crop)
                    else:
                        _in = img
                        if a.derotate:
                            from droneracing.derotate import derotate_img, roll_from_rot
                            _roll = roll_from_rot(ekf.rot(), cfg.cam_tilt)
                            _in = derotate_img(img, _roll)
                        _o = gn(_in)
                        _dets, _pk = decode_refined(*_o), decode_corner_peaks(_o[2], _o[3], thr=a.peak_thr)
                        if a.derotate:
                            _dets, _pk = _rerot(_dets, _pk, _roll, H, W)
                        feat = det_features(_dets, H, W, dev, cam_tilt=cfg.cam_tilt, hfov_deg=a.hfov)
                    peaks = torch.zeros(n, 16, 4, device=dev)
                    for _b, _p in enumerate(_pk):
                        if len(_p):
                            _p = _p[_p[:, 1].argsort(descending=True)[:16]]
                            peaks[_b, : len(_p)] = _p
                else:
                    feat = det_features(gn.detect(img), H, W, dev, cam_tilt=cfg.cam_tilt, hfov_deg=a.hfov)
            if a.rs_comp > 0 and a.sim_det is None:  # each corner back to the frame-centre time (row time x gyro rate)
                from droneracing.rscomp import rs_correct_feat, rs_correct_px
                _tsr, _tcr = math.sin(cfg.cam_tilt), math.cos(cfg.cam_tilt)
                _Rbr = torch.tensor([[0.0, _tsr, _tcr], [-1.0, 0.0, 0.0], [0.0, -_tcr, _tsr]], device=dev)
                _omc = (env.quad.omega + torch.randn_like(env.quad.omega) * 0.03) @ _Rbr  # gyro (noise 0.03 rad/s) in the camera frame
                feat = rs_correct_feat(feat, _omc, H, W, a.hfov, cfg.cam_tilt, a.rs_comp / 1000.0)
                if a.ekf3:
                    _live_r = peaks[..., -1 if peaks.shape[-1] == 5 else 1] > 0
                    _uvr = rs_correct_px(peaks[..., 2:4], _omc, fx, H, a.rs_comp / 1000.0)
                    peaks = torch.cat((peaks[..., :2], torch.where(_live_r[..., None], _uvr, peaks[..., 2:4]), peaks[..., 4:]), -1)
            if a.extr_corr and a.sim_det is None:  # remove the calibrated mount rotation from the detections (homography)
                from droneracing.extrinsic import correct_feat, correct_px, rot_xyz
                if "_E_corr" not in globals():
                    _E_corr = rot_xyz(torch.load(root / a.extr_corr, map_location=dev)["deg"].to(dev).float()[: n])
                feat = correct_feat(feat, _E_corr, H, W, a.hfov, cfg.cam_tilt)
                if a.ekf3:
                    # u, v sit at 2:4 in both layouts: mr peaks (slot, type, u, v, score), GateNet peaks (type, score, u, v)
                    _live = peaks[..., -1 if peaks.shape[-1] == 5 else 1] > 0
                    _uvc = correct_px(peaks[..., 2:4], _E_corr, fx, W / 2, H / 2)
                    peaks = torch.cat((peaks[..., :2], torch.where(_live[..., None], _uvc, peaks[..., 2:4]), peaks[..., 4:]), -1)
            if a.min_det_dist > 0 and a.sim_det is None and a.ekf:
                _mcd, _mnd, _lastd = map_gates()
                _gest = _mcd[:, 0] + ekf.dg_for(env.gate_idx, _lastd, _mcd.shape[1])[:, 0]
                _near = (_gest - ekf.p).norm(dim=-1) < a.min_det_dist
                if _near.any():
                    feat = feat * (~_near)[:, None].float()
                    if a.ekf3:
                        peaks = peaks * (~_near)[:, None, None].float()
            if a.band_check and a.sim_det is None and a.ekf3:
                from droneracing.framecheck import corrupt_rows
                _rows = corrupt_rows(img)  # [n,H]
                if _rows.any():
                    _ri = lambda v: v.round().long().clamp(0, H - 1)  # noqa: E731
                    _ar = torch.arange(n, device=dev)
                    _pin = _rows[_ar[:, None], _ri(peaks[..., 3])]  # peaks (type, score, u, v)
                    peaks = peaks * (~_pin)[..., None].float()
                    _fd = feat.reshape(n, 4, -1).clone()
                    _cy = ((_fd[..., 9:16:2] + 1.0) * 0.5 * H)  # corner v of the whole-gate detections
                    _din = _rows[_ar[:, None, None], _ri(_cy)].any(-1)
                    _fd[_din] = 0.0
                    feat = _fd.reshape(n, -1)
            if a.frame_check and a.sim_det is None:
                from droneracing.framecheck import corrupt_frames
                _bad = corrupt_frames(img)
                if _bad.any():
                    _empty = det_features([torch.zeros(0, 9)] * n, H, W, dev, cam_tilt=cfg.cam_tilt, hfov_deg=a.hfov)
                    feat = torch.where(_bad[:, None], _empty, feat)
                    if a.ekf3:
                        peaks = peaks * (~_bad)[:, None, None].float()
            if a.cam_hz > 0 and a.sim_det is None:  # frame rate below the policy rate: no detection on the steps between two frames
                _per = max(1, round(1.0 / (cfg.dt * cfg.decimation) / a.cam_hz))
                if (not _frame_now(step)) if a.cam_hz_exact else (step % _per != 0):
                    feat = det_features([torch.zeros(0, 9)] * n, H, W, dev, cam_tilt=cfg.cam_tilt, hfov_deg=a.hfov)
                    if a.ekf3:
                        peaks = torch.zeros_like(peaks)
            if a.ekf3 and a.sim_det is None and ((_frame_now(step) if a.cam_hz_exact else step % max(1, round(1.0 / (cfg.dt * cfg.decimation) / a.cam_hz)) == 0) if a.cam_hz > 0 else True):
                # perception in the loop, same rule as scripts/percep_breakdown.py: next gate with >= 2 true corners in the image (true camera,
                # mount error included), found when >= 2 of them have a corner peak of the same type within 4 px (before the delay queue)
                _gpx, _gv, _gsv = rig.gate_corners(env)
                _gp = _gpx[:, 1]
                if a.extr_corr:  # the peaks were moved to the nominal camera by the calibrated mount rotation: compare in the same camera
                    _gp = correct_px(_gp, _E_corr, fx, W / 2, H / 2)
                _inimg = _gv[:, 1] & (_gp[..., 0] >= 0) & (_gp[..., 0] < W) & (_gp[..., 1] >= 0) & (_gp[..., 1] < H)
                _elig = _gsv[:, 1] & (_inimg.sum(-1) >= 2)
                _pkl = peaks[..., 1] > 0 if peaks.shape[-1] == 4 else peaks[..., 4] > 0
                _d = (peaks[:, None, :, 2:4] - _gp[:, :, None]).norm(dim=-1)  # [n,4 corners,16 peaks]
                _same = peaks[:, None, :, 0].long() == torch.arange(4, device=dev)[None, :, None] if peaks.shape[-1] == 4 else peaks[:, None, :, 1].long() % 4 == torch.arange(4, device=dev)[None, :, None]
                _hit = ((_d < 4.0) & _pkl[:, None] & _same).any(-1) & _inimg
                _hit_any = ((_d < 4.0) & _pkl[:, None]).any(-1) & _inimg
                _ext = (_gp.amax(1) - _gp.amin(1)).amax(-1)
                _gc, _, _ = env._gate_frame(torch.minimum(env.gate_idx, env._last_slot()))
                _gd = (_gc - env.quad.pos).norm(dim=-1)
                _pl = globals().setdefault("_percep_loop", {"elig": [], "found": [], "found_any": [], "ext": [], "dist": []})
                _pl["elig"].append(_elig.cpu()); _pl["found"].append((_hit.sum(-1) >= 2).cpu()); _pl["found_any"].append((_hit_any.sum(-1) >= 2).cpu()); _pl["ext"].append(_ext.cpu())
                _pl["dist"].append(_gd.cpu())
                _pl.setdefault("rej", []).append((_bad if a.frame_check else torch.zeros(n, dtype=torch.bool, device=dev)).cpu())
                if a.percep_log:  # whole-gate detections (what the EKF base update uses) against the true next gate
                    _fdl = feat.reshape(n, 4, -1)
                    _dpx = (_fdl[..., 8:16].reshape(n, 4, 4, 2) + 1.0) * 0.5 * torch.tensor([W, H], device=dev)
                    _best = torch.full((n,), 1e6, device=dev)
                    for _perm in ([0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2], [3, 2, 1, 0], [2, 1, 0, 3], [1, 0, 3, 2], [0, 3, 2, 1]):
                        _dd = (_dpx[:, :, _perm] - _gp[:, None]).norm(dim=-1).mean(-1)  # [n,4 dets]
                        _dd = torch.where(_fdl[..., 0] > 0, _dd, torch.full_like(_dd, 1e6))
                        _best = torch.minimum(_best, _dd.amin(-1))
                    _c0l, _n0l, _ = env._gate_frame(torch.minimum(env.gate_idx, env._last_slot()))
                    _cam_ax = quat_to_rot_(env.quad.quat)[:, :, 0] * math.cos(cfg.cam_tilt) + quat_to_rot_(env.quad.quat)[:, :, 2] * math.sin(cfg.cam_tilt)
                    _tog = torch.nn.functional.normalize(_c0l - env.quad.pos, dim=-1)
                    _view = torch.rad2deg(torch.acos((_tog * _n0l).sum(-1).abs().clamp(max=1.0)))  # obliqueness of the gate plane to the line of sight
                    _off = torch.rad2deg(torch.acos((_tog * _cam_ax).sum(-1).clamp(-1, 1)))  # angle of the gate from the optical axis
                    _row = torch.stack((_elig.float(), _ext * (410.0 / W), _inimg.sum(-1).float(), _view, _off, env.quad.omega.norm(dim=-1), _gd,
                                        (_best < 8.0 * W / 820).float(), _best * (410.0 / W), (_hit.sum(-1) >= 2).float()), -1)
                    globals().setdefault("_plog", []).append(_row.cpu())
            if a.cam_delay > 0 and a.sim_det is None:
                dq.append((feat, peaks if a.ekf3 else None))
                feat, _pk_d = dq[0]
                if a.ekf3:
                    peaks = _pk_d
            if a.save_frames and step in (5, 150):
                try:
                    _i = 0
                    _idx = int(env.gate_idx[_i])
                    _cw = env.gate_pos[_i, _idx]
                    _o = rig.objs_rt["s1_bar0"].data.root_pos_w[_i]
                    print(f"DBG step {step}: gate_idx {_idx} gate_pos {[round(float(v), 2) for v in _cw]} origin {[round(float(v), 1) for v in rig.origins[_i]]} "
                          f"bar0 sim pos {[round(float(v), 2) for v in _o]} quad pos {[round(float(v), 2) for v in env.quad.pos[_i]]} "
                          f"cam pos_w {[round(float(v), 2) for v in rig.cam.data.pos_w[_i]]} gate_size {float(env.gate_size[_i]):.2f} n_gates {int(env.n_gates[_i])}", flush=True)
                except Exception as _e:  # noqa: BLE001
                    print("DBG failed:", _e)
            if a.save_frames and step in (5, 150, 400):
                try:
                    import numpy as _np
                    from PIL import Image as _Im
                    _r = rgb[:8].detach().float()
                    print(f"frame stats step {step}: dtype {rgb.dtype} shape {tuple(rgb.shape)} min {float(_r.min()):.3f} max {float(_r.max()):.3f} mean {float(_r.mean()):.3f}", flush=True)
                    _x = (_r / (255.0 if float(_r.max()) > 1.5 else 1.0)).clamp(0, 1) * 255
                    _x = _x.byte()
                    _x = _x.permute(0, 2, 3, 1) if _x.shape[1] == 3 else _x
                    _m = _np.concatenate([_x[i].cpu().numpy() for i in range(len(_x))], 1)
                    Path(root / a.save_frames).mkdir(parents=True, exist_ok=True)
                    _Im.fromarray(_m).save(root / a.save_frames / f"frames_{step}.png")
                except Exception as _e:  # noqa: BLE001
                    print("save-frames failed:", _e)
            if a.diag:
                from droneracing.quad import quat_to_rot as _q2r
                Rw = _q2r(env.quad.quat)
                c0, n0, _ = env._gate_frame(torch.minimum(env.gate_idx, env._last_slot()))
                rel0 = (Rw.transpose(1, 2) @ (c0 - env.quad.pos).unsqueeze(-1)).squeeze(-1)
                n0b = (Rw.transpose(1, 2) @ n0.unsqueeze(-1)).squeeze(-1)
                uv, zc, okp = project_corners(env, H, W, a.hfov)
                u0, z0 = uv[:, 1], zc[:, 1]
                ctr = u0.mean(1)
                vis0 = okp[:, 1] & (ctr[:, 0] >= 0) & (ctr[:, 0] < W) & (ctr[:, 1] >= 0) & (ctr[:, 1] < H) & (z0.min(-1).values > 0.3) \
                    & (((u0[..., 0] >= 0) & (u0[..., 0] < W) & (u0[..., 1] >= 0) & (u0[..., 1] < H)).sum(-1) >= 2)
                fd = feat.reshape(n, 4, -1)
                pos_est = fd[..., 1:4] * 5.0
                valid = fd[..., 0] > 0
                err_all = (pos_est - rel0[:, None]).norm(dim=-1)
                err_all = torch.where(valid, err_all, torch.full_like(err_all, 1e6))
                best = err_all.min(-1)
                kb = err_all.argmin(-1)
                nest = fd[torch.arange(n), kb, 4:7]
                diag["vis"].append(vis0.cpu()); diag["found"].append((best.values < 3.0).cpu()); diag["err"].append(best.values.cpu())
                diag["dist"].append(rel0.norm(dim=-1).cpu())
                dpx = (fd[..., 8:16].reshape(n, 4, 4, 2) + 1) * 0.5 * torch.tensor([W, H], device=dev)  # detected corners in px
                cerr_all = (dpx - u0[:, None]).norm(dim=-1).mean(-1)
                cerr_all = torch.where(valid, cerr_all, torch.full_like(cerr_all, 1e6))
                diag["cerr"].append(cerr_all.min(-1).values.cpu())
                diag["ext"].append((u0.amax(1) - u0.amin(1)).amax(-1).cpu())
                diag["nin"].append(((u0[..., 0] >= 0) & (u0[..., 0] < W) & (u0[..., 1] >= 0) & (u0[..., 1] < H)).sum(-1).cpu())
                diag["nerr"].append((torch.acos((nest * n0b).sum(-1).clamp(-1, 1)) * 57.3).cpu())
            imu = env.imu_obs()
            if a.dump:
                _R = __import__("droneracing.quad", fromlist=["quat_to_rot"]).quat_to_rot(env.quad.quat)
                _Rt = _R.transpose(1, 2)
                _gl = []
                for _j in range(2):
                    _idx = torch.minimum(env.gate_idx + _j, env._last_slot())
                    _c, _nrm, _ = env._gate_frame(_idx)
                    _gl.append(torch.cat(((_Rt @ (_c - env.quad.pos).unsqueeze(-1)).squeeze(-1) / 5.0, (_Rt @ _nrm.unsqueeze(-1)).squeeze(-1),
                                          env.vis_seen[env.arange, _idx].float()[:, None]), -1))
                for _k, _v in zip(("feat", "imu", "act", "vel", "up", "gate"),
                                  (feat, imu, env.prev_action.clone(), (_Rt @ env.quad.vel.unsqueeze(-1)).squeeze(-1) / 10.0, _R[:, 2, :], torch.stack(_gl, 1))):
                    buf[_k].append(_v.cpu())
                buf["valid"].append((since_reset >= 3).cpu())
            if a.ekf:
                ekf.predict(imu, dt)
                mc, mn, last_ = map_gates()
                if a.dump:  # offline replay of the filter: map gates, gate index and ground truth at this step
                    for _k, _v in (('mc', mc), ('mn', mn), ('gi', env.gate_idx), ('last', last_), ('gtp', env.quad.pos), ('gtv', env.quad.vel), ('gtq', env.quad.quat)):
                        buf.setdefault(_k, []).append(_v.cpu().clone())
                if a.ekf3 and args_cli.decoy_prev:  # ignore detections of the gate just passed (split-S / double gates)
                    _pi = (env.gate_idx - 1).clamp_min(0)
                    _pc, _pn, _ = env._gate_frame(_pi, noisy=True)
                    feat_u, peaks_u = ekf.mask_decoys(feat, peaks, env.gate_idx, mc, mn, _pc, _pn)
                    ekf.update(feat_u, env.gate_idx, mc, mn, last_, peaks=peaks_u)
                elif a.ekf5:
                    ekf.update(feat, env.gate_idx, mc, mn, last_, peaks=peaks, delay=(a.cam_delay if a.ekf_delay < 0 else a.ekf_delay))
                elif a.ekf3:
                    ekf.update(feat, env.gate_idx, mc, mn, last_, peaks=peaks)
                else:
                    ekf.update(feat, env.gate_idx, mc, mn, last_)
                if cfg.age_obs:  # an accepted PnP measurement changes the per-gate correction ekf.dg of exactly that gate
                    if dg_prev is None:
                        dg_prev = ekf.dg.clone()
                    age_t[:, :ekf.dg.shape[1]] = (age_t[:, :ekf.dg.shape[1]] + dt / 3.0).clamp_max(1.0)
                    age_t[:, :ekf.dg.shape[1]][(ekf.dg != dg_prev).any(-1)] = 0.0
                    dg_prev = ekf.dg.clone()
                vb_, up, rel_, nrm = ekf.outputs(env.gate_idx, last_, mc, mn, n_out=cfg.n_obs_gates)
                vel, rel = vb_ / 10.0, rel_ / 5.0
                seen_l = torch.ones(n, cfg.n_obs_gates, device=dev)
            else:
                x = torch.cat((feat, imu, env.prev_action, strap.step(imu)), -1)[:, None]
                o, h = est(x, h)
                vel, up, rel, nrm, seen_l = split_out(o[:, 0])
            if a.dump:
                buf["est"].append(torch.cat((vel, up, rel.reshape(n, -1), nrm.reshape(n, -1), seen_l), -1).cpu())
                if a.ekf:
                    _Rt_ = quat_to_rot_(env.quad.quat)
                    _Re_ = ekf.rot()
                    _yaw_t = torch.atan2(_Rt_[:, 1, 0], _Rt_[:, 0, 0]); _yaw_e = torch.atan2(_Re_[:, 1, 0], _Re_[:, 0, 0])
                    _dy = torch.atan2(torch.sin(_yaw_e - _yaw_t), torch.cos(_yaw_e - _yaw_t)) * 57.3
                    buf["dbg"].append(torch.stack(((ekf.p - env.quad.pos).norm(dim=-1), _dy, ekf.accepted, ekf.P[:, 0:3, 0:3].diagonal(dim1=1, dim2=2).sum(-1).sqrt(),
                                                   ekf.P[:, 3:6, 3:6].diagonal(dim1=1, dim2=2).sum(-1).sqrt(), (ekf.ba.norm(dim=-1)), (ekf.bg.norm(dim=-1)),
                                                   (ekf.v - env.quad.vel).norm(dim=-1)), -1).cpu())
            flag = (seen_l > 0).float()[..., None]  # [n,2,1]
            parts = [vel, up, imu[:, :3], env.prev_action]
            for j in range(cfg.n_obs_gates):
                parts += [rel[:, j] * flag[:, j], nrm[:, j] * flag[:, j], env.gate_size[:, None] * flag[:, j], flag[:, j]]
            pol_obs = torch.cat(parts + [hist.reshape(n, -1)], -1)
            if cfg.age_obs:
                _ar = torch.arange(n, device=dev)
                pol_obs = torch.cat((pol_obs, torch.stack([age_t[_ar, torch.minimum(env.gate_idx + _j, last_).clamp(max=63)] for _j in range(cfg.n_obs_gates)], -1)), -1)
            for ch in filter(None, a.ablate.split(",")):
                sl = {"vel": slice(0, 3), "up": slice(3, 6), "gate": slice(13, 29)}[ch]
                if a.ablate_gt:
                    _R = quat_to_rot_(env.quad.quat); _Rt = _R.transpose(1, 2)
                    if ch == "vel":
                        pol_obs[:, sl] = (_Rt @ env.quad.vel.unsqueeze(-1)).squeeze(-1) / 10.0
                    elif ch == "up":
                        pol_obs[:, sl] = _R[:, 2, :]
                    else:
                        _gp = []
                        for _j in range(cfg.n_obs_gates):
                            _idx = torch.minimum(env.gate_idx + _j, env._last_slot())
                            _c, _nrm, _ = env._gate_frame(_idx)
                            _gp += [(_Rt @ (_c - env.quad.pos).unsqueeze(-1)).squeeze(-1) / 5.0, (_Rt @ _nrm.unsqueeze(-1)).squeeze(-1),
                                    env.gate_size[:, None], torch.ones(n, 1, device=dev)]
                        pol_obs[:, sl] = torch.cat(_gp[:8], -1)
                else:
                    pol_obs[:, sl] = obs[:, sl]
            if a.opp_head:  # camera opponents: detections (u, v, r px, full frame) -> body-frame position via the known size, world-frame tracker for the velocity
                _isf = (_frame_now(step) if a.cam_hz_exact else step % max(1, round(1.0 / dt / a.cam_hz)) == 0) if a.cam_hz > 0 else True
                _ff = 0.5 * W / math.tan(math.radians(a.hfov) / 2)
                _ts_, _tc_ = math.sin(cfg.cam_tilt), math.cos(cfg.cam_tilt)
                _Rbo = torch.tensor([[0.0, _ts_, _tc_], [-1.0, 0.0, 0.0], [0.0, -_tc_, _ts_]], device=dev)
                _Rw = ekf.rot()
                opp_feat = torch.zeros(n, cfg.opp_obs * 7, device=dev)
                for _i in range(n):
                    _trk = opp_tracks[_i]
                    for _t in _trk:
                        _t["p"] = _t["p"] + _t["v"] * dt
                        _t["age"] += dt
                    if _isf and len(_opp_dets[_i]):
                        for _d in _opp_dets[_i]:
                            _z = _ff * 0.16 / float(_d[2].clamp_min(1.0))
                            _pc = torch.stack(((_d[0] - W / 2) / _ff * _z, (_d[1] - H / 2) / _ff * _z, torch.tensor(_z, device=dev)))
                            _rb = _pc @ _Rbo.T + torch.tensor([0.1, 0.0, 0.03], device=dev)
                            _pw = ekf.p[_i] + _Rw[_i] @ _rb
                            _best, _bd = None, 3.0
                            for _t in _trk:
                                _dd = float((_t["p"] - _pw).norm())
                                if _dd < _bd:
                                    _best, _bd = _t, _dd
                            if _best is None:
                                _trk.append({"p": _pw, "v": ekf.v[_i].clone(), "age": 0.0, "pl": _pw.clone(), "tl": step * dt, "hits": 1})
                            else:
                                _tt = step * dt - _best["tl"]
                                if 0 < _tt < 0.5:
                                    _best["v"] = 0.6 * _best["v"] + 0.4 * (_pw - _best["pl"]) / _tt
                                _best["p"], _best["pl"], _best["tl"], _best["age"] = _pw, _pw.clone(), step * dt, 0.0
                                _best["hits"] += 1
                    opp_tracks[_i] = _trk = [_t for _t in _trk if _t["age"] < 0.4]
                    _trk.sort(key=lambda _t: float((_t["p"] - ekf.p[_i]).norm()))
                    _trk_u = [_t for _t in _trk if _t["hits"] >= a.opp_hits]
                    if env.opp is not None and not bool(seen[_i]):  # diagnostics against the true opponents
                        _od_ = (env.opp.pos[_i] - env.quad.pos[_i]).norm(dim=-1) + (~env.opp.active[_i]).float() * 1e3
                        opp_stats["steps"] = opp_stats.get("steps", 0) + 1
                        opp_stats["near"] = opp_stats.get("near", 0) + int((_od_ < 12).any())
                        opp_stats["tr_used"] = opp_stats.get("tr_used", 0) + int(len(_trk_u) > 0)
                        opp_stats["near_used"] = opp_stats.get("near_used", 0) + int((_od_ < 12).any() and len(_trk_u) > 0)
                        opp_stats["fp_used"] = opp_stats.get("fp_used", 0) + int(any(float((env.opp.pos[_i] - _t["p"]).norm(dim=-1).min()) > 3.0 for _t in _trk_u))
                    for _k, _t in enumerate(_trk_u[: cfg.opp_obs]):
                        _relb = _Rw[_i].T @ (_t["p"] - ekf.p[_i])
                        _rvb = _Rw[_i].T @ (_t["v"] - ekf.v[_i])
                        opp_feat[_i, _k * 7:(_k + 1) * 7] = torch.cat((torch.ones(1, device=dev), (_relb / 10.0).clamp(-3, 3), (_rvb / 10.0).clamp(-3, 3)))
                pol_obs = torch.cat((pol_obs, opp_feat), -1)
                if env.opp is not None:  # ground truth bookkeeping: opponents in view vs. tracks
                    opp_stats["frames"] += int(_isf) * n
            elif env.opp is not None:  # opponent block at the END of the policy observation (as in RaceEnv / EkfLoopEnv)
                pol_obs = torch.cat((pol_obs, env.opp.obs(True)), -1)
            hist = torch.roll(hist, -1, dims=1)
            hist[:, -1] = torch.cat((vel, imu[:, :3], env.prev_action), -1)
            R = __import__("droneracing.quad", fromlist=["quat_to_rot"]).quat_to_rot(env.quad.quat)
            vb = (R.transpose(1, 2) @ env.quad.vel.unsqueeze(-1)).squeeze(-1) / 10.0
            alive = (~seen).float()
            vel_err += float(((vel - vb).norm(dim=-1) * 10 * alive).sum())
            up_err += float((torch.acos((torch.nn.functional.normalize(up, dim=-1) * R[:, 2, :]).sum(-1).clamp(-1, 1)) * 57.3 * alive).sum())
            steps_alive += float(alive.sum())
            # closed-loop error of the next-gate position the policy sees, by true distance (compare with the torch sensor model)
            _c0, _, _ = env._gate_frame(torch.minimum(env.gate_idx, env._last_slot()))
            _rt = (R.transpose(1, 2) @ (_c0 - env.quad.pos).unsqueeze(-1)).squeeze(-1)
            _ge, _gd = (rel[:, 0] * 5.0 - _rt).norm(dim=-1), _rt.norm(dim=-1)
            if a.video:
                capture(img, rel, nrm, up, R, step)
            for _bi, (_lo, _hi) in enumerate(((0, 2), (2, 4), (4, 8), (8, 99))):
                _m = (alive > 0) & (_gd >= _lo) & (_gd < _hi)
                gerr_bins[_bi].append(_ge[_m].cpu())
        act =ac.actor(ac.pn(pol_obs)).clamp(-1, 1)
        if prev_act is not None:
            rec["jitter"] += (~seen) * (act - prev_act).abs().mean(-1)
        prev_act = act
        obs, _, _, _, done, info = env.step(act)
        speed_sum += env.quad.vel.norm(dim=-1) * (~seen).float()
        rig.update(env)
        if a.video:
            for _k in range(_VN):
                if bool(done[_k]):
                    end_episode(_k, bool(info["finished"][_k]), bool(info["crashed"][_k]), int(info["gates_passed"][_k]), int(info["n_gates"][_k]))
        if a.opp_head and done.any():
            for _i in torch.nonzero(done).flatten().tolist():
                opp_tracks[_i] = []
        if a.ekf and done.any():
            ekf_reset(torch.nonzero(done).flatten())
            age_t[done] = 1.0
            if dg_prev is not None:
                dg_prev[done] = ekf.dg[done]
        if h is not None and done.any():
            h = h * (~done).float()[None, :, None]
            strap.reset(done)
            hist = hist * (~done).float()[:, None, None]
        if a.dump:
            buf["done"].append(done.cpu())
            for _k2 in ("finished", "crashed", "gates_passed"):  # outcome of the episode that ends at this step (read it where done is set)
                buf.setdefault(_k2, []).append(info[_k2].clone().cpu())
            since_reset = torch.where(done, torch.zeros_like(since_reset), since_reset + 1)
        new = done & ~seen
        if new.any():
            _cc = info["cause"][new & info["crashed"]]
            for _c in _cc.tolist():
                _causes[int(_c)] = _causes.get(int(_c), 0) + 1
            rec["fin"][new] = info["finished"][new].float()
            rec["crash"][new] = info["crashed"][new].float()
            rec["len"][new] = info["ep_len"][new].float() * dt
            rec["twr"][new] = info["twr"][new]
            rec["tlen"][new] = info["track_len"][new]
            rec["mspeed"][new] = speed_sum[new] / info["ep_len"][new].float().clamp_min(1.0)
            rec["gates"][new] = info["gates_passed"][new].float()
            rec["ng"][new] = info["n_gates"][new].float()
            seen |= new
        if seen.all() and not a.dump:
            break
rec["jitter"] = rec["jitter"] / max(1, step)
r = {k: v.cpu() for k, v in rec.items()}
print(f"episodes finished {int(seen.sum())}/{n} in {step+1} steps", flush=True)
if a.opp_head:
    print("OPP CAMERA STATS", opp_stats, flush=True)
print(f"CRASH CAUSES {_causes}  (1 frame, 2 pole, 4 ground, 5 out, 6 opponent)", flush=True)
summarize(r, "ORACLE OBS" if a.oracle_obs else f"VISION STACK ablate={a.ablate or 'none'} det={'gatenet' if a.sim_det is None else 'sim'+str(a.sim_det)} {'EKF' if a.ekf else 'GRU'}")
if not a.oracle_obs and steps_alive:
    print(f"estimator in closed loop: velocity err {vel_err/steps_alive:.2f} m/s, attitude err {up_err/steps_alive:.1f} deg | render blank {rh_blank/max(1, rh_n)*100:.1f}% frozen {rh_frozen/max(1, rh_n)*100:.1f}%", flush=True)
if not a.oracle_obs and gerr_bins[0] is not None and any(len(b) for b in gerr_bins):
    _parts = []
    for (_lo, _hi), _b in zip(((0, 2), (2, 4), (4, 8), (8, 99)), gerr_bins):
        _e = torch.cat(_b) if _b else torch.zeros(0)
        if len(_e):
            _parts.append(f"{_lo}-{_hi}m med {float(_e.median()):.2f} p90 {float(_e.quantile(0.9)) if len(_e) < 16_000_000 else float('nan'):.2f}")
    print("gate obs err (estimator in closed loop): " + " | ".join(_parts), flush=True)
if a.diag and diag["vis"]:
    vis, found, err, dist, nerr = (torch.cat(diag[k]) for k in ("vis", "found", "err", "dist", "nerr"))
    print(f"gate in analytic FOV in {float(vis.float().mean())*100:.0f}% of frames; GateNet found it in {float((found & vis).float().sum()/vis.float().sum())*100:.0f}%", flush=True)
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 20)):
        m = vis & found & (dist >= lo) & (dist < hi)
        mv = vis & (dist >= lo) & (dist < hi)
        if m.any():
            print(f"  dist {lo}-{hi} m: detect {float(m.float().sum()/mv.float().sum().clamp_min(1))*100:.0f}% | PnP pos err mean {err[m].mean():.2f} median {err[m].median():.2f} m | normal err median {nerr[m].median():.1f} deg", flush=True)
if a.diag and diag["vis"]:
    cerr, ext, nin = (torch.cat(diag[k]) for k in ("cerr", "ext", "nin"))
    ok_ = vis & (cerr < 40)
    for name, m0 in (("all 4 corners in image", nin == 4), ("2-3 corners in image", (nin >= 2) & (nin < 4))):
        for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 400)):
            m = ok_ & m0 & (ext >= lo) & (ext < hi)
            if m.sum() > 20:
                print(f"  corner err px | {name} | extent {lo}-{hi}px: mean {cerr[m].mean():.2f} median {cerr[m].median():.2f} p90 {cerr[m].quantile(0.9):.2f} (n {int(m.sum())})", flush=True)
if "_percep_loop" in globals() and _percep_loop["elig"]:
    _e, _f, _fa, _x = (torch.cat(_percep_loop[k]) for k in ("elig", "found", "found_any", "ext"))
    _x = _x * (410.0 / W)  # gate size in 410-px-wide image units (comparable between 410x308 and 820x616 lines)
    print(f"PERCEPTION in loop: next gate with >= 2 corners in view in {int(_e.sum())} frames; found (>= 2 corners, 4 px) {100 * float((_f & _e).sum() / _e.sum().clamp_min(1)):.1f} %"
          f" (any corner type {100 * float((_fa & _e).sum() / _e.sum().clamp_min(1)):.1f} %)", flush=True)
    for lo, hi in ((0, 15), (15, 30), (30, 60), (60, 120), (120, 2000)):
        _m = _e & (_x >= lo) & (_x < hi)
        if _m.sum() > 20:
            print(f"  PERCEPTION gate size {lo}-{hi} px(410 scale): found {100 * float((_f & _m).sum() / _m.sum()):.1f} % (n {int(_m.sum())})", flush=True)
    _dd = torch.cat(_percep_loop["dist"])
    for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 10), (10, 100)):
        _m = _e & (_dd >= lo) & (_dd < hi)
        if _m.sum() > 20:
            print(f"  PERCEPTION gate distance {lo}-{hi} m: found {100 * float((_f & _m).sum() / _m.sum()):.1f} % (n {int(_m.sum())})", flush=True)
    _m = _e & (_dd >= 1.0)
    print(f"PERCEPTION in loop, gates at >= 1 m (KAIST distance gate): found {100 * float((_f & _m).sum() / _m.sum().clamp_min(1)):.1f} % (n {int(_m.sum())})", flush=True)
    _rj = torch.cat(_percep_loop["rej"])
    _m2 = _m & ~_rj
    print(f"PERCEPTION in loop, gates at >= 1 m, frames not rejected as corrupt: found {100 * float((_f & _m2).sum() / _m2.sum().clamp_min(1)):.1f} % (n {int(_m2.sum())}, rejected frames {100 * float((_rj & _m).sum() / _m.sum().clamp_min(1)):.1f} %)", flush=True)
if a.percep_log and "_plog" in globals():
    _lp = root / a.percep_log
    _lp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cols": ["elig", "size410", "nin", "view_deg", "off_axis_deg", "omega", "dist", "found_box", "box_err410", "found_corners"], "rows": torch.cat(_plog)}, _lp)
    print("PERCEP LOG", len(torch.cat(_plog)), flush=True)
if a.save_crops and "_saved_crops" in globals():
    _od = root / a.save_crops
    _od.mkdir(parents=True, exist_ok=True)
    _sv2 = {k: torch.cat(v) for k, v in _saved_crops.items()}
    _sv2.update({"hfov": a.hfov, "hw": (a.crop, a.crop), "full_hw": (H, W), "tok": torch.zeros(len(_sv2["img"]), 11)})
    torch.save(_sv2, _od / f"data_{a.seed}.pt")
    print("SAVED CROPS", len(_sv2["img"]), flush=True)
if a.dump:
    out = root / a.dump
    out.mkdir(parents=True, exist_ok=True)
    _sv = {k: torch.stack(v) for k, v in buf.items()}
    if rig.extr_err is not None:  # true mount rotation per drone (true camera = nominal * E), for checking a calibration
        _sv["extr_err"] = rig.extr_err.cpu()
    torch.save(_sv, out / "data.pt")
    print("DUMP DONE", flush=True)
app.close()
