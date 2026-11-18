# Release v3 FAST: soup85 (2026-11-05)

Fast model: **soup85 = 0.15 x g37 (safe) + 0.85 x g38d (fast)**, weight interpolation, no extra training (`recipe/make_soup.py`). Detector g17, estimator e4, calibration `calib/cal15_41.pt`.
User goal: mean speed 20 m/s (MonoRace level) with success above 75 %.

## Result (Isaac target line, detector g17, 4 drones per seed)
| Set | Success | True mean speed | Lap |
|---|---|---|---|
| soup85, seeds 71-78 | 28/32 = 87.5 % (95 % 72-95) | 20.59 m/s | 17.40 s |
| soup85, FRESH seeds 79-86 (not used to choose anything) | 25/32 = 78.1 % (95 % 61-89) | 20.66 m/s | 17.31 s |
| soup85, FRESH seeds 87-94 (third run) | 30/32 = 93.8 % (95 % 80-98) | 20.39 m/s | ~17.4 s |
| **soup85, 24 seeds pooled, 96 flights** | **83/96 = 86.5 %** (95 % **78.2-91.9**) | **20.55 m/s** | 17.4 s |
| g38d (fast ingredient), 16 seeds, 96 flights | 72/96 = 75.0 % (95 % 65.5-82.6) | 21.1 m/s | 17.1 s |
| g38d on the same fresh seeds 79-86 | 24/32 = 75.0 % | 21.09 m/s | 17.04 s |

`mean-speed` = real mean speed of the finished flights (VISION line of isaac/eval_vision.py). `path-speed` = reference track length / lap time (18.9-19.0 m/s for soup85, reads ~8 % low).
**Both parts of the goal are met: true mean speed 20.55 m/s (>= 20) and success 86.5 % over 96 flights, with the lower end of the 95 % interval (78.2 %) above 75 %.** Difference to g38d (75.0 % over 96 flights): +11.5 points (z about 2.0).

## Why this model
Surrogate (success / path-speed, `logs/surr_soup.txt`): g37 94.0/15.7, soup50 90.1/17.4, soup70 87.0/18.5, **soup85 86.5/19.1**, g38d 84.1/19.4. soup85 was chosen from the surrogate before any Isaac run (path-speed ~19.1). Weight interpolation gives a smooth speed-success path between the safe and the fast policy; eight reward and curriculum changes before it did not beat g38d in Isaac.
Fallbacks: `policy/g38d_fast_ingredient` (21.1 m/s, 75.0 %), `policy/g37_safe_ingredient` (15.5 m/s path-speed, 93.8 %).

## Open
- Isaac only; the pipeline has not run on hardware (Jetson speed test, mount calibration below 0.5 deg, delay measurement: see `releases/v2_INTERIM_g34_g17_nov3/MANIFEST.md` and `jetson/README.md`).
- Try soup90 / soup95 for more speed, soup70 for more margin: the speed-success path is smooth in the surrogate, confirm in Isaac.
- Isaac success follows the filter velocity error (g37 0.35, g38d 0.45): a better velocity estimate would shift the whole path.
