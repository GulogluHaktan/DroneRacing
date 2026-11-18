# INTERIM snapshot v2: g34 policy + g17 detector (2026-11-03)

**NOT a final release.** Target (> 85 % flight success) is not reached. Saved only as a fallback; the final release comes after the crash analysis and the policy polish.

Safe model, own camera line. Best measured so far: **53/64 = 82.8 %** flight success over 16 seeds (4 drones per seed), full target line.
Standard error of this number is about 4.7 points (64 flights). Do not read differences of a few points between candidates as real.
The speed-limited training is intentional: this is the "safe" model. A fast model is the next step (see the end).

## Contents
| Path | What |
|---|---|
| `policy/g34/` | actor (model.pt + config.json + log.csv). Main policy. |
| `policy/g32/`, `policy/g27/` | its ancestors (g27 -> g32 -> g34) |
| `gatenet/g17_gatenet.pt`, `gatenet/gatenet_g17_384.onnx` | GateNet detector, 384x384 crop; ONNX for TensorRT (not yet timed on the Jetson) |
| `estimator/estimator.pt` | e4 estimator used by `eval_vision.py --estimator` |
| `calib/cal15_41.pt` | per-drone mount rotation estimate (offline calibration, 0.2-0.6 deg residual) |
| `eval/` | per-seed Isaac results (t34g17_cal = final, t34g16*, t34 = older detectors) |
| `recipe/` | every script that made the model, with the exact flags |
| `logs/` | training logs of g34, g16, g17 |
| `code_snapshot.tgz` | droneracing/, isaac/, scripts/ at release time |

## Lineage (how the model was trained)
1. **Policy**: g27 = g17 policy + MonoRace perception penalty (`--perc-pen 2`), see `recipe/run_g27_target.sh`.
   g32 = g27 + realistic detection model in training (`recipe/run_g32.sh`).
   **g34** = g32 + 1000 iterations in the estimator-in-the-loop env (`--ekf-loop`: actor sees GateEKF5 fed by simulated detections, own camera 30 Hz, 50 ms delay), `--perc-pen 2` (`recipe/run_g34.sh`).
2. **Detector**: g15 -> g16 (fine-tuned on d9 + d10 + d11 crops flown by g34, `recipe/run_onpolicy16.sh`) -> **g17** (g16 + d12 crops flown by g34 with g16, lr 1e-4, `recipe/run_g17.sh`).
   Detector training augmentation: `droneracing/realcam_aug.py` (rolling shutter, motion blur, short exposure, lens edge, corrupt bands) + roll derotation with 3 deg error. No mount-error augmentation (it is a camera rotation, handled by calibration).
3. **Training conditions**: profile `own_drone`, 4096 envs, `--v-max 13` (speed limit), `--finish-time 75`, obs history 8, 3 gates in the observation, split noise 3 2 1.5 1.5, fixed-prob 1 / jitter 0.3 0.05. Full flag line in `recipe/run_g34.sh`.

## Evaluation line (all flags in `recipe/run_g17.sh`)
Isaac, 820x616 render, 30 Hz camera, `--cam-delay 5` (50 ms), `--ekf5`, rolling shutter 20 ms, 10 % corrupt frames, exposure aug 0.4, `--tilt-err 3` (unknown 3 deg mount error, `--tilt-seed 7`) with `--extr-corr calib/cal15_41.pt`.
Seeds 21-30, 35, 36, 41-44. Seeds 31-34, 51-54, 61-64 are detector TRAINING seeds: never evaluate on them.

| Candidate (policy g34) | 16-seed result |
|---|---|
| g15 detector | 76.6 % |
| g16 | 79.7 % |
| **g17** | **82.8 %** (seeds 21-28: 28/32; fresh seeds: 25/32) |
| g18 (one more on-policy round) | worse on 8 seeds (26/32 vs 28/32), dropped |

## Known limits
- Same seed gives different results between runs (seed 36: 1/4 in the eval, 4/4 in a dump run, same config). Pool flights over seeds.
- Surrogate (`eval_state.py --ekf-loop`) reads too high for policies trained inside it (g34 ~91 % surrogate vs ~80 % Isaac). Rank by Isaac.
- Real drone: mount error (1/2/3 deg unknown -> 50/33/17 %), delay mismatch (20 ms halves success). Calibrate offline below 0.5 deg and time-stamp frames. Nothing here ran on hardware.
- g37 (g34 + 1000 more iterations, `recipe/run_g37.sh`) is being evaluated; it is not part of this release yet.

## Next: fast model
Train with the speed limit removed from the reward/limits (`--v-max` higher, no speed penalty), starting from g34, in the same estimator-in-the-loop env. Keep this release as the fallback.
