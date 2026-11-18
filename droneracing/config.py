"""Configuration dataclasses for the Phase A (state-based) racing environment."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VehicleRanges:
    """Randomization ranges for the vehicle. SI units.

    ``dr_scale`` in [0, 1] shrinks every range around its (geometric/arithmetic)
    centre: 0 = nominal vehicle, 1 = full ranges. Used for the speed-vs-DR-range study.
    """

    mass: tuple[float, float] = (0.25, 1.2)  # [kg], log-uniform
    max_thrust: tuple[float, float] = (5.0, 45.0)  # [N] total, log-uniform, then clamped to twr_range
    twr_range: tuple[float, float] = (1.8, 7.0)  # thrust-to-weight limits (keeps vehicle controllable)
    arm: tuple[float, float] = (0.06, 0.16)  # [m] centre-to-motor distance ("size"), uniform
    drag_cd: tuple[float, float] = (0.5, 1.2)  # [-] drag coefficient, uniform
    tau_rate: tuple[float, float] = (0.02, 0.05)  # [s] body-rate loop time constant, uniform
    tau_thrust: tuple[float, float] = (0.02, 0.06)  # [s] motor/thrust lag, uniform
    max_rate: float = 10.0  # [rad/s] commanded roll/pitch rate at |action|=1
    yaw_rate_scale: float = 0.5
    direct_motor: bool = False  # action = 4 motor commands (MonoRace G&CNet style, no rate loop / mixer); False = thrust + body-rate command (CTBR)
    dr_scale: float = 1.0


@dataclass
class TrackRules:
    """Rules of the procedural track generator (all lengths [m], angles [rad])."""

    min_gates: int = 4
    max_gates: int = 10
    seg_len: tuple[float, float] = (3.0, 8.0)  # distance between consecutive gates
    turn_max: tuple[float, float] = (0.5, 1.7)  # max heading change per gate; scaled by vehicle capability
    max_climb: float = 0.5  # max climb/descent angle of a segment
    z_range: tuple[float, float] = (0.8, 3.0)  # gate centre height
    clearance: float = 3.0  # min distance between non-adjacent gates
    gate_size: tuple[float, float] = (0.8, 1.5)  # inner opening side
    gate_frame: float = 0.25  # frame thickness (outside the opening)
    max_obstacles: int = 6
    obs_radius: tuple[float, float] = (0.10, 0.25)
    obs_lateral: tuple[float, float] = (0.3, 1.8)  # offset of an obstacle from the straight gate-gate line
    obs_gate_clear: float = 1.5  # min distance obstacle <-> gate centre (xy), beyond radius
    obs_spacing: float = 1.5  # min distance between obstacle axes
    obs_height: float = 3.5  # poles span ground to this height (cannot be flown over)
    start_dist: tuple[float, float] = (1.5, 3.0)  # start distance before gate 0
    n_retries: int = 8
    # --- markers / decisions (used when EnvCfg.markers is True) ---
    turn_thresh: float = 0.35  # |heading change| above this -> L/R marker
    climb_thresh: float = 0.6  # |height change| above this [m] -> U/D marker
    fork_prob: float = 0.0  # probability that a track contains one fork (correct gate + decoy)
    fork_offset: tuple[float, float] = (2.2, 3.5)  # lateral distance between the two gates of a fork
    land_dist: float = 4.0  # landing point distance after the last gate
    # --- aerobatic modules (stacked gates; 0 = off) ---
    split_prob: float = 0.0  # per-gate chance of a split-S: upper gate, then lower gate beneath it, opposite direction
    split_gap: tuple[float, float] = (0.3, 1.3)  # clear gap between the two stacked frames (spacing = size + 2*frame + gap)
    split_lat: float = 0.35  # max lateral misalignment of the lower gate [m]
    split_yaw: float = 0.12  # max yaw misalignment of the lower gate [rad]
    loop_prob: float = 0.0  # per-gate chance of a loop: lower gate, upper gate (inverted, opposite direction), lower gate again
    loop_dz: tuple[float, float] = (4.4, 5.6)  # loop diameter (vertical spacing)
    ladder_prob: float = 0.0  # per-gate chance of a ladder (A2RL 2026 'precision climb and descent'; geometry assumed): 3 gates in line, each step up (or down)
    ladder_step: tuple[float, float] = (0.8, 1.3)  # height change per rung [m]
    ladder_dx: tuple[float, float] = (3.0, 5.0)  # horizontal spacing of the rungs [m]
    turnu_thresh: float = 2.6  # |heading change| above this -> U-turn marker
    walk_max_gates: int = 0  # random-walk tracks use at most this many gates (0 = max_gates); lets max_gates grow for multi-lap hall tracks
    # --- A2RL-style hall tracks: closed lap inside a rectangular hall, flown for 1-2 laps (0 = off) ---
    hall_prob: float = 0.0  # fraction of tracks that are hall laps (the rest: random walk / drag)
    hall_w: tuple[float, float] = (50.0, 100.0)  # hall length [m] (A2RL hall: 100 x 30 m; one lap ~175 m, 11 gates, ~16 m apart)
    hall_h: tuple[float, float] = (18.0, 30.0)  # hall width [m]
    hall_gates_per_lap: tuple[int, int] = (8, 11)  # A2RL: 11 gates per lap
    hall_laps: tuple[int, int] = (1, 2)  # A2RL: 2 laps (the same gates are passed again)
    hall_shape: tuple[float, float] = (2.0, 3.5)  # superellipse exponent: 2 = ellipse, larger = longer straights, tighter corners
    hall_margin: float = 3.0  # keep gates this far from the walls
    hall_wavy: float = 0.15  # radial perturbation of the lap outline (fraction)
    hall_z_amp: float = 0.9  # height variation along the lap [m]
    # --- replica of the A2RL 2025 final track (track.a2rl_2025_lap), 2 laps; 0 = off ---
    fixed_prob: float = 0.0
    fixed_jitter: tuple[float, float] = (0.0, 0.0)  # gate position [m] (z gets 0.3x) and yaw [rad] jitter per episode
    fixed_scale: tuple[float, float] = (1.0, 1.0)  # whole-layout x/y scale
    fixed_mirror: float = 0.0  # probability of the mirrored layout
    # --- drag-race style straights (long run-up, the only place 20+ m/s is reachable) ---
    drag_prob: float = 0.0  # fraction of random-walk tracks that are near-straight corridors
    drag_seg: tuple[float, float] = (12.0, 25.0)
    drag_gates: tuple[int, int] = (4, 7)


@dataclass
class RewardCfg:
    progress: float = 1.0  # per metre of distance reduction to the target gate
    gate: float = 5.0
    finish: float = 20.0
    crash: float = 10.0
    rate: float = 0.0005  # per (rad/s)^2
    action_diff: float = 0.01
    yaw_cmd: float = 0.0  # per step: penalty * (yaw-rate command)^2  (stops yaw chatter / spinning)
    yaw_diff: float = 0.0  # per step: penalty * (change of the yaw-rate command)^2
    alive: float = 0.0
    finish_time: float = 0.0  # extra finish bonus * remaining-time fraction (rewards faster laps)
    decoy: float = 10.0  # penalty for flying through the wrong gate of a fork
    land_slow: float = 0.05  # per step: penalty * speed^2 near the landing point
    perc: float = 0.0  # per step: keep the next gate in the camera view, perc * exp(-angle^4)  (Swift / MonoRace)
    perc_pen: float = 0.0  # per step penalty, perc_pen * (angle - 60 deg) when the next gate is more than 60 deg off the optical axis (MonoRace)
    speed_pen: float = 0.0  # per second: -speed_pen * max(0, |v| - speed_cap) (safe model: slower, more margin for perception errors)
    speed_cap: float = 14.0  # [m/s]
    slow_pen: float = 0.0  # per step (arg is per second): -slow_pen * max(0, slow_min - |v|), after slow_grace s from the start (fast model: pays for being slow)
    slow_min: float = 22.0  # [m/s]
    opp_near: float = 0.0  # per step: -opp_near * (1 - gap/opp_safe)^2 for an opponent closer than opp_safe (gap = surface distance)
    slow_grace: float = 1.5  # [s] no slow penalty while the drone leaves the standstill
    align: float = 0.0  # on gate pass: -align * max(0, angle(velocity, gate normal) - align_free) / 45 deg (straight gate entry; crashes at the 25 deg gate 4 of the replica)
    align_free: float = 10.0  # [deg] angle without penalty
    center: float = 0.0  # on gate pass: center * (1 - normalised offset from the gate centre)  (MonoRace)
    gate_speed: float = 0.0  # on gate pass: gate_speed * min(speed, gate_speed_cap)/10  (pay for speed AT the gate, not only for finishing fast)
    gate_speed_cap: float = 25.0
    soft_frame: float = 3.0  # penalty for a gate-frame hit in a soft-collision episode (the hit counts as a pass without bonuses, no termination)
    v_max: float = 0.0  # progress per step is capped at v_max * dt (MonoRace speed/safety knob); 0 = off


@dataclass
class SensorCfg:
    """Sensing / actuation imperfections. Every magnitude is multiplied by ``scale`` (0 = perfect)."""

    scale: float = 0.0
    # state-estimate errors seen by the actor (the critic always gets the truth)
    vel_noise: float = 0.12  # [m/s] per step
    vel_bias: float = 0.15  # [m/s] per episode
    vel_drift: float = 0.0  # [m/s/sqrt(s)] per-axis random-walk velocity error growing with time since reset (closed-loop estimator drift)
    vel_drift_cap: float = 3.0  # [m/s] per-axis clamp of that drift
    vel_noise_rel: float = 0.03  # velocity error proportional to speed (VIO scale error)
    att_noise: float = 0.015  # [rad] per step
    att_bias: float = 0.03  # [rad] per episode (misalignment / estimator offset)
    gyro_noise_obs: float = 0.05  # [rad/s] on the policy's rate observation
    gate_noise: float = 0.04  # [m] + gate_noise_dist * distance, per step
    gate_noise_dist: float = 0.015
    bias_rand: tuple[float, float] = (1.0, 1.0)  # (lo, hi) log-uniform per-episode multiplier of gate / velocity biases (1,1 = off)
    gate_err_corr: float = 0.93  # AR(1) coefficient of the gate-estimate error while the gate is in view (vis_obs)
    gate_bias: float = 0.12  # [m] per gate per episode (track map is never exact)
    gate_yaw_bias: float = 0.04  # [rad] per gate per episode
    obstacle_noise: float = 0.10  # [m]
    odo_tau: float = 0.0  # >0 (map_obs): pose error of the filter is an Ornstein-Uhlenbeck process with this time constant [s] and std odo_sigma
    odo_sigma: float = 0.8  # [m] stationary std of that pose error per axis
    yaw_noise: float = 0.03  # [rad] per-sighting gate-normal error (map_obs; PnP normal)
    odo_drift: float = 0.04  # [m/sqrt(s)] odometry random-walk drift of the held gate map (vis_obs)
    fov_h: float = 1.08  # [rad] half horizontal field of view used by the visibility test (62 deg)
    fov_v: float = 0.96  # [rad] half vertical field of view (55 deg)
    see_range: float = 18.0  # [m] max distance a gate is detected
    gate_dropout: float = 0.02  # per-step chance that the gate observation freezes (stale)
    gate_dropout_hold: float = 0.8  # chance the freeze continues next step
    obs_delay: tuple[int, int] = (0, 3)  # perception latency in policy steps (50 Hz -> 0..60 ms)
    act_delay: tuple[int, int] = (0, 4)  # command latency in physics sub-steps (5 ms each): NN inference + serial + FC, 0..20 ms
    # IMU
    gyro_scale_err: float = 0.01
    accel_scale_err: float = 0.01
    accel_sat_g: float = 16.0  # accelerometer range (A2RL drone saturates at +-16 g)
    gyro_sat: float = 34.9  # [rad/s] (2000 deg/s)
    vib_accel: float = 0.8  # [m/s^2] vibration noise at full thrust
    vib_gyro: float = 0.15  # [rad/s]
    sat_glitch: float = 15.0  # [m/s^2] inconsistent error std once the accelerometer is near saturation
    # actuation / environment
    sag: float = 0.12  # max battery-sag thrust loss at the end of an episode
    wind: float = 0.8  # [m/s^2] constant disturbance magnitude (per episode)
    gust: float = 0.8  # [m/s^2] OU gust std
    gust_tau: float = 0.5  # [s]


@dataclass
class EnvCfg:
    n_envs: int = 2048
    device: str = "cuda"
    dt: float = 0.005  # physics step [s]
    decimation: int = 4  # policy at 1/(dt*decimation) = 50 Hz
    gate_shrink: float = 0.0  # training only: extra shrink of the half opening [m] (MonoRace trains with 0.4-0.55 m openings); set by train_state, never read from saved args
    gate_box: bool = False  # oriented box instead of the sphere: the extent of the body (half x,y = arm/sqrt2 + prop radius, half z) projected on the gate's lateral and vertical axes with the attitude at the crossing
    box_prop: float = 0.075  # prop radius + margin [m]
    box_hz: float = 0.05  # half height of the body [m]
    gate_body: bool = False  # gate passage checks the drone body (centre must be within half opening - drone radius) instead of a point
    cam_period: int = 1  # camera sightings only every cam_period policy steps (frame rate < policy rate)
    corner_vis_p: tuple = (0.9, 0.14, 0.06)  # detection probability with 4 / 3 / 2 corners in the image (default: GateNet g3 measurement)
    corner_vis: bool = False  # a sighting needs the gate corners in the image (partial gates are rarely detected, tiny gates are missed), instead of just the centre in the FOV
    tail_prob: float = 0.0  # per-episode, per-gate probability of a heavy-tail perception event (measured p99/median error ratio ~15-30)
    tail_mult: float = 1.0  # error scale of such a gate (map bias and sighting noise)
    # Swift-style empirical residual model: the policy's gate / velocity / attitude errors are replayed from a bank of
    # closed-loop error traces (Isaac camera line now, real flights later), chosen by distance to the next gate. "" = off.
    resid_bank: str = ""
    resid_scale: float = 1.0
    resid_prob: float = 1.0  # fraction of envs that replay the bank (rest: parametric errors)
    split_noise: tuple = ()  # sighting-noise multipliers for the split-S lower gate and the gates after it (camera dumps: x6 then ~x3); () = off
    age_obs: bool = False  # append the time since the last sighting of each observed gate (scaled to 0..1 over 3 s) to the policy obs
    max_episode_s: float = 40.0
    n_obs_gates: int = 2
    n_obs_obstacles: int = 3
    markers: bool = False  # marker tokens in obs (teacher), forks and landing phase
    landing: bool = False  # after the last gate: land (slow + low) inside the landing zone
    land_radius: float = 1.5
    land_speed: float = 1.0
    land_height: float = 0.35
    gyro_noise: float = 0.02  # [rad/s]
    gyro_bias: float = 0.01
    accel_noise: float = 0.15  # [m/s^2]
    accel_bias: float = 0.05
    policy_sees_params: bool = True  # False: vehicle params hidden from the actor (student-like)
    map_obs: bool = False  # approximate track map known from the start (gate centre/yaw error = gate_bias / gate_yaw_bias); a sighting replaces it by a PnP measurement
    vis_obs: bool = False  # camera-like gate knowledge: only gates seen so far (FOV test), world estimate held + odometry drift
    obs_history: int = 0  # K past (body velocity, gyro, previous action) frames appended to the actor obs (system identification cue)
    out_of_bounds: float = 60.0  # [m] from gate 0
    omega_limit: float = 0.0  # [rad/s] body rate above which the episode counts as a crash (MonoRace: 1700 deg/s); 0 = off
    soft_frac: float = 0.0  # fraction of episodes with soft gate-frame collisions (penalty, no termination; two-phase training of 2512.09571)
    cam_tilt: float = 0.75  # [rad] camera up-tilt about the body y axis (A2RL drone: 43-50 deg)
    mid_init_prob: float = 0.0  # fraction of resets that start in front of a random gate with some speed
    mid_init_dist: tuple[float, float] = (1.5, 6.0)  # [m] distance in front of that gate
    mid_init_speed: tuple[float, float] = (0.0, 10.0)  # [m/s] along the gate normal
    mid_focus: tuple = ()  # gate indices that a mid-track start prefers (fast model: the gates where fast flights crash)
    mid_focus_prob: float = 0.0  # share of the mid-track starts that use a focus gate
    opp_max: int = 0  # scripted opponent drones per episode (1..opp_max, chosen per episode); 0 = single-drone racing (see droneracing/opponents.py)
    opp_prob: float = 1.0  # share of the episodes that have opponents
    opp_speed: tuple[float, float] = (6.0, 13.0)  # [m/s] opponent cruise speed range
    opp_radius: float = 0.15  # [m] opponent body radius (contact = our crash)
    opp_obs: int = 2  # nearest opponents in the policy observation (appended at the END: old policies grow zero-initialised input columns)
    opp_range: float = 25.0  # [m] detection range
    opp_detect: float = 0.9  # per-step detection probability inside the field of view
    opp_noise: tuple[float, float, float] = (0.10, 0.04, 0.5)  # position sigma = a + b * distance [m]; velocity sigma c [m/s]
    opp_safe: float = 1.0  # [m] surface gap below which the proximity penalty starts
    vehicle: VehicleRanges = field(default_factory=VehicleRanges)
    track: TrackRules = field(default_factory=TrackRules)
    reward: RewardCfg = field(default_factory=RewardCfg)
    sensors: SensorCfg = field(default_factory=SensorCfg)


def racing_vehicle() -> VehicleRanges:
    """Race-class quads (roughly 3-5 inch): high thrust-to-weight, fast rate loop."""
    return VehicleRanges(
        mass=(0.30, 0.90), max_thrust=(12.0, 60.0), twr_range=(3.5, 8.0), arm=(0.07, 0.13),
        drag_cd=(0.6, 1.2), tau_rate=(0.015, 0.035), tau_thrust=(0.015, 0.04), max_rate=12.0,
    )


def racing_track() -> TrackRules:
    """Longer straights, gentler turns: room to actually build speed."""
    return TrackRules(
        seg_len=(6.0, 16.0), turn_max=(0.4, 1.3), clearance=5.0, z_range=(0.8, 3.5),
        obs_lateral=(0.4, 2.0), obs_spacing=2.5,
    )


def a2rl_vehicle() -> VehicleRanges:
    """Ranges centred on the A2RL x DCL 2025 organizer drone (5.1-inch, 2207 / 2100 KV, 6S).

    Nominal values from MonoRace (arXiv 2601.15222) and the KAIST team paper (2512.20475):
    mass 0.966 kg, k_w*4*w_max^2 -> ~59.6 m/s^2 max specific thrust (TWR ~6.1, 57.6 N total),
    motor tau 0.025 s, frame 30x28x10 cm. Ranges are about +-20 % (thrust +-25 %).
    """
    return VehicleRanges(
        mass=(0.80, 1.16), max_thrust=(44.0, 72.0), twr_range=(4.0, 8.0), arm=(0.105, 0.145),
        drag_cd=(0.35, 0.70), tau_rate=(0.012, 0.030), tau_thrust=(0.015, 0.040), max_rate=14.0,
    )


def own_vehicle() -> VehicleRanges:
    """Ranges centred on OUR drone (Manta 5 DeadCat, Velox 2207 1950 KV, HQProp 5.1x3.1x3, 6S 1550 mAh, ~0.885 kg).

    Narrow domain randomization around the planned parts (docs: vault droneracing-own-drone-model-spec). Nominals are
    estimates until the bench test: mass 0.885 kg, thrust 50 N, tau_thrust 0.027 s. Ranges are +-12 % (thrust +-20 %, the
    largest unknown); replace by the measured value +-12 % after the bench.
    """
    return VehicleRanges(
        mass=(0.78, 0.99), max_thrust=(40.0, 60.0), twr_range=(4.0, 7.5), arm=(0.108, 0.118),
        drag_cd=(0.40, 0.60), tau_rate=(0.016, 0.025), tau_thrust=(0.020, 0.034), max_rate=14.0,
    )


def a2rl_track() -> TrackRules:
    """A2RL x DCL style: 1.5 m square gates, ~16 m spacing (about 175 m per lap, 11 gates, two laps), tight turns."""
    return TrackRules(
        min_gates=6, max_gates=11, seg_len=(5.0, 11.0), turn_max=(0.5, 1.6), clearance=4.0,
        z_range=(1.45, 3.8), gate_size=(1.48, 1.52), gate_frame=0.6, split_gap=(0.05, 0.30),
        obs_lateral=(0.4, 2.0), obs_spacing=2.5,  # A2RL gate: 1.5 m opening, 2.7 m outer (0.6 m frame); double gate = stacked frames
    )


def a2rl_long_track() -> TrackRules:
    """Longer, more open A2RL-like layout: 8-12 gates, 6-15 m between gates, gentler heading changes (higher top speed)."""
    r = a2rl_track()
    r.min_gates, r.max_gates = 8, 12
    r.seg_len = (6.0, 15.0)
    r.turn_max = (0.4, 1.3)
    r.clearance = 5.0
    return r


def a2rl_hall_track() -> TrackRules:
    """Mixture closer to the real event: hall-bounded closed laps (1-2 laps, 8-11 gates per lap, walls = crash),
    drag-race straights and the old random-walk tracks with split-S modules. Vehicle/track randomization stays on."""
    r = a2rl_long_track()
    r.max_gates = 22
    r.walk_max_gates = 12
    r.hall_prob = 0.5
    r.drag_prob = 0.15
    return r


def make_cfg(args: dict, n_envs: int, policy_sees_params: bool | None = None) -> EnvCfg:
    """Single place that turns run args (CLI or saved config.json) into an EnvCfg.

    Default profile is ``a2rl`` (A2RL organizer drone + A2RL-style track, randomized around it).
    Old runs without a ``profile`` key are treated as ``wide`` for backward compatibility.
    """
    profile = args.get("profile", "wide")
    cfg = EnvCfg(n_envs=n_envs)
    cfg.policy_sees_params = (not args.get("hide_params", False)) if policy_sees_params is None else policy_sees_params
    a2rl_like = profile in ("a2rl", "a2rl_long", "a2rl_hall", "own_drone")
    if profile == "own_drone":  # our parts, narrow vehicle DR, fixed (not randomized) camera model: IMX219-160, 30 Hz
        cfg.vehicle, cfg.track = own_vehicle(), a2rl_hall_track()
        cfg.out_of_bounds = 160.0
        cfg.cam_tilt = 0.785  # 45 deg up; the +-5 deg mount error is a per-drone evaluation option (--tilt-err), not a training range
        cfg.sensors.fov_h, cfg.sensors.fov_v = 1.12, 0.84  # half angles of the ~128 x 96 deg field (est.; measure with Kalibr)
        cfg.sensors.obs_delay = (2, 7)  # 20-70 ms at 100 Hz: image latency 40-100 ms (est.) with NO compensation in the filter
        if args.get("obs_delay"):  # with GateEKF5 (delay compensation) the policy sees a much smaller residual delay
            cfg.sensors.obs_delay = (int(args["obs_delay"][0]), int(args["obs_delay"][1]))
        if not args.get("cam_hz"):
            args = {**args, "cam_hz": 30.0}
    elif profile == "a2rl_hall":
        cfg.vehicle, cfg.track = a2rl_vehicle(), a2rl_hall_track()
        cfg.out_of_bounds = 160.0
    elif profile == "a2rl_long":
        cfg.vehicle, cfg.track = a2rl_vehicle(), a2rl_long_track()
        cfg.out_of_bounds = 160.0  # tracks reach 110 m of path; the old 60 m radius around gate 0 flagged valid flights as crashes
    elif profile == "a2rl":
        cfg.vehicle, cfg.track = a2rl_vehicle(), a2rl_track()
    elif profile == "racing":
        cfg.vehicle, cfg.track = racing_vehicle(), racing_track()
    cfg.vehicle.dr_scale = args.get("dr_scale", 1.0)
    cfg.obs_history = int(args.get("obs_history", 0))
    cfg.map_obs = bool(args.get("map_obs", False))
    if args.get("n_obs_gates"):
        cfg.n_obs_gates = int(args["n_obs_gates"])
    cfg.vis_obs = bool(args.get("vis_obs", False)) or cfg.map_obs
    if args.get("sensor_profile") == "vision":  # error levels measured on the GRU estimator (scripts/train_estimator.py, e3)
        s = cfg.sensors
        s.vel_noise, s.vel_bias, s.vel_noise_rel = 0.20, 0.30, 0.04
        s.gate_noise, s.gate_noise_dist, s.gate_bias, s.gate_yaw_bias = 0.05, 0.04, 0.08, 0.06
        s.att_bias, s.att_noise = 0.03, 0.02
    if args.get("sensor_profile") == "ekf":  # map-based EKF + GateNet g3 in Isaac (provisional values from the first closed-loop runs; see scripts/error_stats.py)
        s = cfg.sensors
        s.vel_noise, s.vel_bias, s.vel_noise_rel = 0.25, 0.35, 0.03  # velocity error ~0.9 m/s, bounded
        s.att_bias, s.att_noise = 0.03, 0.02  # attitude error ~2.8 deg
        s.gate_noise, s.gate_noise_dist, s.gate_bias, s.gate_yaw_bias = 0.06, 0.02, 0.25, 0.09  # PnP noise when seen; map error (bias) when not
        s.yaw_noise = 0.25
        s.odo_tau, s.odo_sigma = 3.0, 0.7  # bounded pose error of the filter
    if args.get("sensor_profile") == "closedloop":  # error levels measured with the policy flying on the GRU estimator + real GateNet in Isaac (scripts/error_stats.py)
        s = cfg.sensors
        s.vel_noise, s.vel_bias, s.vel_noise_rel = 0.20, 0.30, 0.04
        s.vel_drift = 0.45  # 0.7 m/s after 1-2 s, 1.4 at 2-5 s, 2.1 at 5-10 s, 3.2 at 10-20 s
        s.gate_noise, s.gate_noise_dist, s.gate_bias, s.gate_yaw_bias = 0.10, 0.12, 0.30, 0.35  # median 1 m at 5 m, normal error median ~20 deg
        s.att_bias, s.att_noise = 0.05, 0.03
    cfg.vehicle.direct_motor = bool(args.get("motor_direct", False))
    if cfg.vehicle.direct_motor:
        cfg.omega_limit = 30.0
    if args.get("no_obstacles", False):
        cfg.track.max_obstacles = 0
    cfg.reward.finish_time = args.get("finish_time", 0.0)
    if args.get("bias_rand"):
        cfg.sensors.bias_rand = (float(args["bias_rand"][0]), float(args["bias_rand"][1]))
    cfg.reward.gate_speed = float(args.get("gate_speed", 0.0))
    cfg.reward.yaw_cmd = float(args.get("yaw_cmd", 0.0))
    cfg.reward.yaw_diff = float(args.get("yaw_diff", 0.0))
    sscale = args.get("sensor_scale")
    cfg.sensors.scale = (1.0 if a2rl_like else 0.0) if sscale is None else float(sscale)
    if a2rl_like:  # reward shaping from the A2RL / Swift literature (see docs/phase_b_spec.md)
        cfg.reward.center, cfg.reward.perc, cfg.reward.rate = 1.0, 0.02, 0.001
        cfg.mid_init_prob = 0.5
    if args.get("mid_prob") is not None:
        cfg.mid_init_prob = float(args["mid_prob"])
    if args.get("mid_speed"):
        cfg.mid_init_speed = (float(args["mid_speed"][0]), float(args["mid_speed"][1]))
    if args.get("mid_focus"):
        cfg.mid_focus = tuple(int(x) for x in args["mid_focus"])
        cfg.mid_focus_prob = float(args.get("mid_focus_prob") or 0.5)
    # soft_frac is deliberately NOT read from args: saved training args would silently make every evaluation soft; train_state sets it explicitly
    if args.get("v_max") is not None:
        cfg.reward.v_max = float(args["v_max"])
    if args.get("crash_pen") is not None:
        cfg.reward.crash = float(args["crash_pen"])
    if args.get("speed_pen") is not None:
        cfg.reward.speed_pen = float(args["speed_pen"]) * cfg.dt * cfg.decimation  # per-second weight -> per policy step
        cfg.reward.speed_cap = float(args.get("speed_cap") or 14.0)
    if args.get("slow_pen"):
        cfg.reward.slow_pen = float(args["slow_pen"]) * cfg.dt * cfg.decimation  # per-second weight -> per policy step
        cfg.reward.slow_min = float(args.get("slow_min") or 22.0)
        cfg.reward.slow_grace = float(args.get("slow_grace") if args.get("slow_grace") is not None else 1.5)
    if args.get("align_pen") is not None:
        cfg.reward.align = float(args["align_pen"])
    if args.get("center_rew") is not None:
        cfg.reward.center = float(args["center_rew"])
    cfg.markers = bool(args.get("markers", False))
    cfg.landing = bool(args.get("landing", False))
    cfg.track.fork_prob = args.get("fork_prob", cfg.track.fork_prob)
    cfg.track.split_prob = args.get("split_prob", cfg.track.split_prob)
    cfg.track.loop_prob = args.get("loop_prob", cfg.track.loop_prob)
    cfg.track.ladder_prob = args.get("ladder_prob", cfg.track.ladder_prob) or 0.0
    if args.get("hall_prob") is not None:
        cfg.track.hall_prob = float(args["hall_prob"])
    if args.get("drag_prob") is not None:
        cfg.track.drag_prob = float(args["drag_prob"])
    if args.get("fixed_prob"):
        cfg.track.fixed_prob = float(args["fixed_prob"])
        cfg.track.fixed_jitter = tuple(args.get("fixed_jitter") or (0.0, 0.0))
        cfg.track.fixed_scale = tuple(args.get("fixed_scale") or (1.0, 1.0))
        cfg.track.fixed_mirror = float(args.get("fixed_mirror") or 0.0)
        cfg.track.max_gates = max(cfg.track.max_gates, 22)
    hz = args.get("policy_hz")
    if hz:  # policy rate: 50 Hz = default (dt 5 ms x 4); per-step quantities are rescaled so the time behaviour stays comparable
        hz = float(hz)
        cfg.dt = 0.005 if hz <= 200 else 0.002
        cfg.decimation = max(1, round(1.0 / (hz * cfg.dt)))
        ratio = cfg.dt * cfg.decimation / 0.02
        rw, sn_ = cfg.reward, cfg.sensors
        rw.rate, rw.action_diff, rw.perc, rw.yaw_cmd, rw.yaw_diff = (x * ratio for x in (rw.rate, rw.action_diff, rw.perc, rw.yaw_cmd, rw.yaw_diff))
        if args.get("smooth_mult"):  # fast model: smoother flying (rate and action-change penalties x M) keeps the filter velocity error low
            rw.rate, rw.action_diff = rw.rate * float(args["smooth_mult"]), rw.action_diff * float(args["smooth_mult"])
        sn_.gate_err_corr = sn_.gate_err_corr**ratio  # AR(1) per policy step
        sn_.act_delay = (0, min(sn_.act_delay[1], cfg.decimation))  # command latency is capped at one policy period
    if args.get("perc_pen"):
        cfg.reward.perc_pen = float(args["perc_pen"]) * cfg.dt * cfg.decimation / 0.02  # per-second weight, scaled to the policy step
    cfg.gate_body = bool(args.get("gate_body", False))
    cfg.gate_box = bool(args.get("gate_box", False))
    cfg.corner_vis = bool(args.get("corner_vis", False))
    if args.get("corner_vis_p"):
        cfg.corner_vis_p = tuple(float(x) for x in args["corner_vis_p"])
    cfg.age_obs = bool(args.get("age_obs", False))
    cfg.opp_max = int(args.get("opp_max") or 0)
    if args.get("opp_prob") is not None:
        cfg.opp_prob = float(args["opp_prob"])
    if args.get("opp_speed"):
        cfg.opp_speed = (float(args["opp_speed"][0]), float(args["opp_speed"][1]))
    if args.get("opp_near"):
        cfg.reward.opp_near = float(args["opp_near"])
    cfg.resid_bank = str(args.get("resid_bank") or "")
    cfg.resid_scale = float(args.get("resid_scale") or 1.0)
    cfg.resid_prob = float(args.get("resid_prob") or 1.0)
    if cfg.resid_bank and cfg.resid_prob >= 1.0:  # the recorded traces already contain detection gaps and latency effects
        cfg.sensors.gate_dropout, cfg.sensors.obs_delay = 0.0, (0, 0)
    cfg.tail_prob = float(args.get("tail_prob", 0.0))
    cfg.split_noise = tuple(args.get("split_noise") or ())
    cfg.tail_mult = float(args.get("tail_mult", 1.0))
    cam_hz = args.get("cam_hz")
    if cam_hz:  # camera frame rate below the policy rate: sightings only every round(policy_hz / cam_hz) steps
        cfg.cam_period = max(1, round(1.0 / (cfg.dt * cfg.decimation) / float(cam_hz)))
    return cfg
