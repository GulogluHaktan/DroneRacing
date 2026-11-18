"""Sanity tests for dynamics, track rules and env logic. Run: python tests/test_core.py"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from droneracing import EnvCfg, RaceEnv, TrackRules, VehicleRanges  # noqa: E402
from droneracing.quad import QuadBatch, GRAVITY, quat_to_rot  # noqa: E402
from droneracing.track import generate_tracks, track_length  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def gen(seed=0):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


def test_hover_holds_position():
    # nominal vehicle: with motor-thrust mismatch (dr_scale > 0) an open-loop hover command legitimately drifts
    q = QuadBatch(64, DEV, VehicleRanges(dr_scale=0.0), gen())
    ids = torch.arange(64, device=DEV)
    q.reset(ids, torch.tensor([0.0, 0.0, 2.0], device=DEV).expand(64, 3), torch.zeros(64, device=DEV))
    hover_cmd = (2.0 / q.twr - 1.0)  # c_cmd=1/twr -> a0 = 2c-1
    a = torch.zeros(64, 4, device=DEV)
    a[:, 0] = hover_cmd
    for _ in range(400):
        q.step(a, 0.005)
    assert (q.pos[:, 2] - 2.0).abs().max() < 0.01, (q.pos[:, 2] - 2.0).abs().max()
    assert q.vel.abs().max() < 0.01


def test_free_fall_matches_gravity():
    cfg = VehicleRanges(dr_scale=0.0)
    q = QuadBatch(4, DEV, cfg, gen())
    ids = torch.arange(4, device=DEV)
    q.reset(ids, torch.tensor([0.0, 0.0, 100.0], device=DEV).expand(4, 3), torch.zeros(4, device=DEV))
    q.motor[:] = 0.0
    q.thr[:] = 0.0
    a = torch.full((4, 4), 0.0, device=DEV)
    a[:, 0] = -1.0  # zero thrust command
    for _ in range(100):  # 0.5 s
        q.step(a, 0.005)
    expected = 0.5 * GRAVITY * 0.5**2
    drop = 100.0 - q.pos[:, 2]
    assert torch.allclose(drop, torch.full_like(drop, expected), atol=0.05), drop


def test_rate_tracking_and_unit_quaternion():
    q = QuadBatch(32, DEV, VehicleRanges(), gen(1))
    ids = torch.arange(32, device=DEV)
    q.reset(ids, torch.tensor([0.0, 0.0, 5.0], device=DEV).expand(32, 3), torch.zeros(32, device=DEV))
    a = torch.zeros(32, 4, device=DEV)
    a[:, 0] = 2.0 / q.twr - 1.0
    a[:, 1] = 0.5
    for _ in range(100):
        q.step(a, 0.005)
    assert torch.allclose(q.omega[:, 0], torch.full((32,), 0.5 * 10.0, device=DEV), atol=0.3), q.omega[:, 0]
    assert torch.allclose(q.quat.norm(dim=-1), torch.ones(32, device=DEV), atol=1e-5)
    R = quat_to_rot(q.quat)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, device=DEV).expand(32, 3, 3), atol=1e-4)


def test_track_rules():
    rules = TrackRules()
    n = 2000
    cap = torch.rand(n, device=DEV, generator=gen(3))
    tr = generate_tracks(n, rules, cap, gen(4), DEV)
    ng = tr["n_gates"]
    assert ng.min() >= rules.min_gates and ng.max() <= rules.max_gates
    p = tr["gate_pos"]
    valid = torch.arange(p.shape[1], device=DEV)[None] < ng[:, None]  # landing slot sits at ground height
    assert p[..., 2][valid].min() >= rules.z_range[0] - 1e-4 and p[..., 2][valid].max() <= rules.z_range[1] + 1e-4
    for i in range(0, n, 97):  # spot-check geometric rules per track
        k = int(ng[i])
        seg = (p[i, 1:k] - p[i, : k - 1]).norm(dim=-1)
        assert (seg >= rules.seg_len[0] - 1e-3).all(), seg
        for a in range(k):
            for b in range(a + 2, k):
                assert (p[i, a] - p[i, b]).norm() >= rules.clearance - 1e-3
        on = tr["obs_on"][i]
        xy = tr["obs_xy"][i][on]
        for a in range(len(xy)):
            assert (xy[a] - p[i, :k, :2]).norm(dim=-1).min() >= rules.obs_gate_clear
            for b in range(a + 1, len(xy)):
                assert (xy[a] - xy[b]).norm() >= rules.obs_spacing - 1e-3
    length = track_length(tr)
    assert length.std() > 3.0, "track length should vary"
    print(f"  track length mean {length.mean():.1f} m, min {length.min():.1f}, max {length.max():.1f}; "
          f"gates mean {ng.float().mean():.1f}; obstacles mean {tr['obs_on'].float().sum(-1).mean():.1f}")


def test_gate_pass_and_crash_logic():
    cfg = EnvCfg(n_envs=4, device=DEV)
    cfg.vehicle.dr_scale = 0.0
    env = RaceEnv(cfg, seed=5)
    env.obs_on[:] = False  # no poles for this test
    ids = torch.arange(4, device=DEV)
    # Put the drone 0.05 m before gate 0 centre, moving through it at 5 m/s.
    for i, lateral in enumerate((0.0, 0.0, 0.7, 5.0)):
        yaw = env.gate_yaw[i, 0]
        n = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)))
        r = torch.stack((torch.sin(yaw), -torch.cos(yaw), torch.zeros_like(yaw)))
        env.gate_size[i] = 1.2
        start = env.gate_pos[i, 0] - 0.05 * n + lateral * r
        env.quad.pos[i] = start
        env.quad.vel[i] = 5.0 * n
        env.quad.quat[i] = torch.tensor([1.0, 0, 0, 0], device=DEV)
        env.quad.thr[i] = 1.0 / env.quad.twr[i]
        env.quad.motor[i] = 1.0 / env.quad.twr[i]
        env.gate_idx[i] = 0
        env.steps[i] = 0
        env.max_steps[i] = 1000
    env.quad.vel[1] *= 0.0  # env 1 stays put: no crossing
    env.quad.thr[1] = 1.0 / env.quad.twr[1]
    env.quad.motor[1] = 1.0 / env.quad.twr[1]
    a = torch.zeros(4, 4, device=DEV)
    a[:, 0] = 2.0 / env.quad.twr - 1.0
    _, _, rew, term, done, info = env.step(a)
    gp = info["gates_passed"]
    assert gp[0] == 1, gp  # through the centre -> passed
    assert gp[1] == 0 and not info["crashed"][1]  # no motion -> nothing
    assert gp[2] == 0 and info["crashed"][2], (gp, info["crashed"])  # 0.7 m lateral: outside opening (0.6) but inside frame band (0.85) -> crash
    # lateral 5 m: flies past the frame entirely -> neither pass nor crash
    assert gp[3] == 0 and not info["crashed"][3]


def test_env_random_policy_runs_and_resets():
    cfg = EnvCfg(n_envs=512, device=DEV)
    env = RaceEnv(cfg, seed=7)
    pol, cri = env.compute_obs()
    assert pol.shape == (512, env.policy_dim) and cri.shape == (512, env.critic_dim)
    n_done = 0
    for _ in range(300):
        a = torch.randn(512, 4, device=DEV) * 0.3
        pol, cri, rew, term, done, info = env.step(a)
        n_done += int(done.sum())
        assert torch.isfinite(pol).all() and torch.isfinite(rew).all()
    assert n_done > 0
    print(f"  random policy: {n_done} episodes ended in 300 steps")


def test_markers_fork_landing_imu():
    from droneracing.track import TURN_L, TURN_R, CLIMB_U, CLIMB_D, FORK_L, FORK_R

    rules = TrackRules(fork_prob=1.0)
    n = 1000
    tr = generate_tracks(n, rules, torch.rand(n, device=DEV, generator=gen(8)), gen(9), DEV)
    cnt = tr["n_gates"]
    p, yw = tr["gate_pos"], tr["gate_yaw"]
    # every track with >=3 gates has exactly one fork; marker side matches geometry
    has = tr["decoy_on"].any(-1)
    assert (has == (cnt >= 3)).all()
    for i in range(0, n, 53):
        if not has[i]:
            continue
        j = int(tr["decoy_on"][i].nonzero()[0])
        assert tr["marker_fork"][i, j - 1] in (FORK_L, FORK_R)
        right = torch.stack((torch.sin(yw[i, j]), -torch.cos(yw[i, j]), torch.zeros((), device=DEV)))
        side_right = float(((p[i, j] - tr["decoy_pos"][i, j]) * right).sum()) > 0  # correct gate on the right?
        assert (tr["marker_fork"][i, j - 1] == FORK_R) == side_right
        assert (p[i, j] - tr["decoy_pos"][i, j]).norm() >= rules.fork_offset[0] - 1e-3
    # turn / climb markers agree with geometry
    k = 1
    dpsi = torch.atan2(torch.sin(yw[:, k + 1] - yw[:, k]), torch.cos(yw[:, k + 1] - yw[:, k]))
    ok = cnt > k + 1
    assert (tr["marker_turn"][ok, k] == torch.where(dpsi[ok] > rules.turn_thresh, TURN_L, torch.where(dpsi[ok] < -rules.turn_thresh, TURN_R, 1))).all()
    # landing point after last gate
    b = torch.arange(n, device=DEV)
    assert torch.allclose(p[b, cnt, 2], torch.full((n,), 0.15, device=DEV))

    cfg = EnvCfg(n_envs=8, device=DEV, markers=True, landing=True)
    cfg.track.fork_prob = 1.0
    cfg.vehicle.dr_scale = 0.0
    env = RaceEnv(cfg, seed=11)
    pol, cri = env.compute_obs()
    assert pol.shape[1] == env.policy_dim and cri.shape[1] == env.critic_dim
    assert env.imu_obs().shape == (8, 6)
    env.obs_on[:] = False
    # decoy crash: put drone at the fork gate's decoy
    env.decoy_on[:] = False
    env.n_gates[:] = 5
    for i in range(8):
        env.decoy_on[i, 1] = True
        env.decoy_pos[i, 1] = env.gate_pos[i, 1] + torch.tensor([0.0, 0.0, 0.0], device=DEV)
    env.gate_idx[:] = 1
    yaw = env.gate_yaw[:, 1]
    n_ = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), -1)
    env.gate_size[:] = 1.2
    env.quad.pos[:] = env.gate_pos[:, 1] - 0.05 * n_
    env.quad.vel[:] = 5.0 * n_
    a = torch.zeros(8, 4, device=DEV)
    a[:, 0] = 2.0 / env.quad.twr - 1.0
    _, _, rew, term, done, info = env.step(a)
    assert info["decoy"].all() and info["crashed"].all()
    # landing: after the final gate, a slow low drone inside the zone finishes
    env2 = RaceEnv(cfg, seed=12)
    env2.obs_on[:] = False
    env2.gate_idx[:] = env2.n_gates
    land = env2.gate_pos[env2.arange, env2.n_gates]
    env2.quad.pos[:] = land + torch.tensor([0.0, 0.0, 0.1], device=DEV)
    env2.quad.vel[:] = 0.0
    env2.quad.thr[:] = 1.0 / env2.quad.twr
    _, _, _, _, _, info2 = env2.step(a)
    assert info2["finished"].all(), info2["finished"]
    print("  markers/fork/landing/IMU ok")


def test_aerobatic_modules():
    import math
    from droneracing.config import TrackRules
    from droneracing.track import generate_tracks, TURN_U
    g = torch.Generator().manual_seed(5)
    r = TrackRules(split_prob=0.3, loop_prob=0.3)
    tr = generate_tracks(500, r, torch.rand(500, generator=g), g, "cpu")
    P, Y, N = tr["gate_pos"], tr["gate_yaw"], tr["n_gates"]
    n_u = 0
    for e in range(500):
        for k in range(int(N[e]) - 1):
            if abs(math.remainder(float(Y[e, k + 1] - Y[e, k]), 2 * math.pi)) > 2.9:
                n_u += 1
                assert float((P[e, k + 1, :2] - P[e, k, :2]).norm()) < 0.4  # stacked (small lateral misalignment)
                assert abs(float(P[e, k + 1, 2] - P[e, k, 2])) > 1.5
                assert tr["marker_turn"][e, k] == TURN_U
    assert n_u > 100, n_u
    print("  aerobatic modules ok", n_u)


def test_sensor_imperfections():
    from droneracing.config import make_cfg
    cfg = make_cfg({"profile": "a2rl", "sensor_scale": 1.0, "markers": True, "landing": True, "split_prob": 0.1}, 128)
    cfg.device = "cpu"
    env = RaceEnv(cfg, seed=2)
    pol, crit = env.compute_obs()
    assert torch.isfinite(pol).all() and torch.isfinite(crit).all()
    # actor sees errors, critic clean: re-computing twice must give different actor obs but same critic obs
    pol2, crit2 = env.compute_obs()
    assert (pol - pol2).abs().max() > 1e-4 and (crit - crit2).abs().max() < 1e-6
    # IMU saturates at +-16 g
    env.quad.spec_b[:] = 400.0
    imu = env.imu_obs()
    assert (imu[:, 3:].abs() * 20.0).max() <= cfg.sensors.accel_sat_g * 9.81 + 1e-3
    # rollout with random actions stays finite, delays/dropouts exercised
    assert env.obs_delay.max() > 0 and env.act_delay.max() > 0
    for _ in range(150):
        obs, c, r, term, done, info = env.step(torch.rand(128, 4) * 2 - 1)
        assert torch.isfinite(obs).all() and torch.isfinite(r).all()
    cfg0 = make_cfg({"profile": "a2rl", "sensor_scale": 0.0}, 64)
    cfg0.device = "cpu"
    e0 = RaceEnv(cfg0, seed=2)
    p0, c0 = e0.compute_obs()
    p1, _ = e0.compute_obs()
    assert (p0 - p1).abs().max() < 1e-6  # perfect sensors are deterministic
    print("  sensor imperfections ok")


def test_mid_init_and_rewards():
    from droneracing.config import make_cfg
    cfg = make_cfg({"profile": "a2rl", "sensor_scale": 0.0, "split_prob": 0.3, "loop_prob": 0.1}, 512)
    cfg.device = "cpu"
    env = RaceEnv(cfg, seed=4)
    idx = env.gate_idx
    assert (idx > 0).float().mean() > 0.3  # about half of the resets start mid-track
    gc, nrm, _ = env._gate_frame(torch.minimum(idx, env._last_slot()))
    s = ((env.quad.pos - gc) * nrm).sum(-1)
    assert (s[idx > 0] < 0).all()  # always in front of the target gate
    assert (env.quad.vel.norm(dim=-1)[idx > 0] > 0).float().mean() > 0.8
    # perception reward is bounded, centre bonus only on passes, v_max caps progress
    cfg.reward.v_max = 3.0
    tot = 0.0
    for _ in range(40):
        _, _, r, _, _, _ = env.step(torch.zeros(512, 4))
        assert torch.isfinite(r).all()
        tot += r.mean().item()
    print("  mid-init / rewards ok")


def test_camera_and_student():
    from droneracing.camera import AnalyticCamera, augment
    from droneracing.config import make_cfg
    from droneracing.student import StudentNet
    cfg = make_cfg({"profile": "a2rl", "sensor_scale": 0.0, "markers": True}, 16)
    cfg.device, cfg.mid_init_prob = "cpu", 0.0
    env = RaceEnv(cfg, seed=3)
    cam = AnalyticCamera(env)
    img = cam.render()
    assert img.shape == (16, 3, 48, 64) and torch.isfinite(img).all() and 0 <= img.min() and img.max() <= 1
    # looking straight at gate 0 from 3 m: the frame ring must be visible
    gc, nrm, _ = env._gate_frame(torch.zeros(16, dtype=torch.long))
    env.quad.pos[:] = gc - 3.0 * nrm
    yaw0 = env.gate_yaw[:, 0]
    from droneracing.quad import yaw_to_quat
    env.quad.quat[:] = yaw_to_quat(yaw0)
    cfg.cam_tilt = 0.0
    img = cam.render()
    ring = ((img - torch.tensor([1.0, 0.35, 0.10]).view(1, 3, 1, 1)).abs().sum(1) < 0.15).float().mean()
    assert ring > 0.02, ring
    aug = augment(img, torch.zeros(16, 3), 1.0)
    assert aug.shape == img.shape and torch.isfinite(aug).all()
    net = StudentNet()
    a, aux, h = net(img[:, None], torch.zeros(16, 1, 10))
    assert a.shape == (16, 1, 4) and aux["token"].shape[-1] == 11 and h.shape == (1, 16, 128)
    print("  camera + student ok")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL OK")
