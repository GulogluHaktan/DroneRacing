# Phase B spec: sensor-based racing (RGB + IMU), marker-conditioned decisions

User decisions (2026-10-14): sensors = RGB + IMU; markers = direction cues + gate choice (fork) + finish/landing.

## Task semantics (implemented in the torch core, flags `markers`, `landing`)
| marker | meaning | consequence |
|---|---|---|
| turn L/S/R on gate k | heading change of the NEXT segment | information the policy cannot see before passing gate k |
| climb U/L/D on gate k | height change of the next segment | same |
| fork L/R on gate j-1 | gate j has two openings; pass the one on that side | flying through the decoy = crash (-10) |
| FINISH on last gate | after it, land: speed < 1 m/s, z < 0.35 m, within 1.5 m of the landing point (4 m after the gate) | lap finishes only on landing |

## Sensors
- IMU: gyro (σ 0.02 rad/s, bias σ 0.01) + accelerometer specific force (σ 0.15 m/s², bias σ 0.05), from `env.imu_obs()`.
- Front camera RGB (planned 64x64..96x96, 50 Hz), Isaac Lab TiledCamera. No position/velocity/gate poses for the student.

## Training plan
1. Teacher: privileged state + marker tokens, torch core (fast).
2. Student: RGB + IMU + last action (+ recurrence), trained by DAgger from the teacher in Isaac Lab, then optional RL fine-tune.
3. Isaac scene: per-env origin offsets (>= 150 m apart, camera far clip 40 m), kinematic gate frames / poles / marker panels moved at reset, marker colours / lighting randomized.

## Status
- done: marker/fork/landing/IMU in torch core + tests.
- next: Isaac scene (gates, poles, marker panels), TiledCamera, student network + DAgger.

## Training pipeline status (2026-10-14)

Profile `a2rl` is the default (`droneracing.config.make_cfg`). Reward terms and where they come from:

| Term | Value (a2rl) | Source / reason |
|---|---|---|
| progress (distance to next gate) | 1.0, optional cap `--v-max` | Swift, MonoRace (cap = speed/safety knob) |
| gate pass | 5 | all |
| gate-centre offset bonus | 1.0 | MonoRace lambda_offset |
| perception (next gate inside camera cone, exp(-angle^4), tilt 43 deg) | 0.02 | Swift / MonoRace lambda_perc; needed so a camera student can see gates |
| body-rate penalty | 0.001 | MonoRace lambda_Omega |
| action change penalty | 0.01 | smoothness |
| crash | 10 | all |
| finish / decoy / landing | 20 / 10 / 0.05 | task-specific |

Initialisation: 50 % of resets start in front of a random gate (1.5-6 m, 0-10 m/s along the gate normal),
as MonoRace's uniform initialisation; this is how split-S / loop gates get enough training samples.
Evaluation always starts at the line (`mid_init_prob = 0`).

Curricula (train_state flags): `--sensor-ramp N` (imperfections 0 -> 1 over N its),
`--v-max-start/--v-max` (progress cap raised over the first 60 % of training),
`--dr-scale`, `--split-prob/--loop-prob/--fork-prob`.
Caveat: training-time "gates %" is inflated by mid-track starts; compare runs with eval_state only.

## Student pipeline (built GPU-free, 2026-10-14)
- `droneracing/camera.py`: `AnalyticCamera` ray-casts gates, marker panels (turn colour, climb stripe, fork
  panel, finish), ground and sky into 48x64 RGB; `augment()` = exposure/gain, noise, motion blur,
  corrupted frames. Stand-in for the Isaac TiledCamera (analytic = fast, not photoreal).
- `droneracing/student.py`: CNN + IMU/prev-action MLP + GRU, aux heads (marker token, next-gate position).
- `scripts/train_student.py`: DAgger (beta decays 1 -> 0), teacher labels from privileged clean obs
  (`RaceEnv.teacher_obs`), BPTT over 32-step sequences. Smoke-tested on CPU only.
- Open: marker teacher not trained yet; student evaluation script; Isaac TiledCamera rendering for the
  final fidelity check (camera hfov 155 deg + distortion, rolling shutter not modelled here).

## First Isaac results (2026-10-15)
- Vision rig check (isaac/check_vision_rig.py): gate frames, marker colours, env isolation, reset follow: PASS.
- Teacher t1 (a2rl, markers+landing+fork 0.3+split 0.1, sensors ramped to 1, 1500 its): held-out
  success 57.8 %, crash 42 %, path speed 7.7 m/s (eval_state, line start, sensors on). Weak: speed/crash trade-off open.
- RGB student (48x64, Isaac TiledCamera) with DAgger, runs/s1 (single-batch, 1 step/it) and runs/s2
  (aggregated buffer, 16 updates/it): works while the teacher drives (beta > 0.4: ~70 % finish), 100 % crash once
  beta -> 0; marker-token loss stays ~1.9-2.7 (near prior).
- Diagnostic scripts/diag_student_nopix.py (same DAgger loop, no pixels):
  gate vectors + IMU, no velocity -> 26-37 % finish; + body velocity -> ~62 % finish (= teacher level).
  => the loop is fine; the bottlenecks are (a) velocity observability and (b) reading gates/markers from pixels.
- Suspected for (b): resolution. At 10 m a 0.55 m marker is ~2 px in 64 px / 120 deg; needs larger markers or
  96x128+ input; also pre-train perception supervised (image -> gate vector, token, velocity) before DAgger.

## Pixel RL attempt p3 (2026-10-15) — stopped at it 360 (~5.9 M steps)
Recurrent PPO from 48x64 RGB + IMU, asymmetric critic, aux heads (weight 0.5), 512 envs, a2rl DR, shape-coded
signs with placement jitter. Result: finish 0 %, crash 100 %, gates ~2 of 7-11 (mid-init starts), marker CE stays
at 2.7 (no learning), action std collapsed 0.49 -> 0.14 (premature convergence). Throughput 2.6 k steps/s.
Literature check (MonoRace, arXiv 2601.15222): the A2RL winner does NOT train pixel RL; it uses a separate
segmentation net (GateNet, 384x384 crops) -> corner detection -> PnP + EKF -> state-based policy trained in a
simple sim. Plan: move to a modular pipeline (perception -> estimator -> state policy).

## Camera orientation bug (found 2026-10-15) + A2RL gate redesign
- `TiledCameraCfg.OffsetCfg.rot` in this Isaac Lab build is **xyzw**; `vision_rig.py` passed wxyz, so every earlier
  pixel experiment (s0-s2 DAgger, p1-p3 pixel RL) saw the world through an unintended camera rotation. Fixed and
  verified: projected ground-truth inner corners now sit exactly on the rendered gate openings
  (isaac/collect_percep.py previews). Earlier student/pixel-RL failures are therefore not conclusive.
- Gate = A2RL x DCL: square, 1.5 m opening, 2.7 m outer (0.6 m panel frame), purple printed panel with white logos,
  violet LED strip on the opening edge, single post; double gate = two stacked frames (Novak 2602.01860,
  Bahnam 2601.15222; look from Azhari 2512.20475 Fig. 6). No sign stands above the gate; task markers are printed
  flush on the frame (top bar: turn code / fork side; bottom bar: climb code), with placement jitter.
- Perception data (`isaac/collect_percep.py`) is policy-free: random re-placement in front of random gates.
