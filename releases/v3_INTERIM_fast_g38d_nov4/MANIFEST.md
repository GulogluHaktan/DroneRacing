# INTERIM snapshot v3: fast model g38d (2026-11-04)

**NOT a final release.** Goal: mean speed 20 m/s (MonoRace level) with success above 75 %. Result: speed met (true mean speed 21.1 m/s over 16 seeds), success exactly 75.0 % over 96 flights (not above).

## Result (Isaac target line, detector g17, 4 drones per seed, seeds 71-78, two independent runs)
| Run | Success | Path-speed | True mean speed | Lap |
|---|---|---|---|---|
| t38d_cal | 24/32 = 75.0 % | 19.12 m/s | (not measured) | 17.18 s |
| t38d2_cal | 24/32 = 75.0 % | 19.31 m/s | **21.12 m/s** | 17.02 s |
| t38d3_cal (FRESH seeds 79-86) | 24/32 = 75.0 % | 19.31 m/s | 21.09 m/s | ~17.0 s |
| **Pooled, 3 runs, 16 different seeds** | **72/96 = 75.0 %** (95 % interval 65.5-82.6 %) | 19.2 | ~21.1 | 17.1 s |

`path-speed` = reference track length / lap time (reads ~10 % below the real speed). `mean-speed` = real mean speed of the finished flights.
Fallbacks: `policy/g38c` (24/32 = 75.0 %, true mean speed 19.91 m/s) and the safe model in `releases/v2_INTERIM_g34_g17_nov3` (g34/g37 line, 90 % at 15.5 m/s).

## Lineage (recipe/)
g37 (safe) -> g38 (v-max 30) -> g38b (gate-speed 2, finish-time 300) -> g38c (slow-pen 2/min 22, gate-speed 10, finish-time 600) -> **g38d** (slow-pen 4/min 23, crash-pen 40, init-std 0.15, 1200 iterations). Env: estimator-in-the-loop (`--ekf-loop`), own_drone profile, sensor-scale 0.6, perc-pen 2, 100 Hz policy.

## What did not help (all Isaac, seeds 71-78)
g38e (slow-pen 5), g38f (center-rew 8), g38g (sensor-scale 1.0: 59.4 %, surrogate had said +7), g38h (smooth-mult 4: 68.8 %), detector g19 on g38d (71.9 %). The surrogate (`eval_state.py --ekf-loop`) misleads for policies trained inside it.

## Open
Success above 75 % at this speed needs a lever that shifts the Isaac frontier. Isaac success follows the filter velocity error (g37 0.35 -> 94 %, g38d 0.45 -> 75 %, g38g 0.60 -> 59 %). Crash anatomy: 98 % frame hits with small filter error (gate 6 at 22-28 m/s, gate 0 at the start).
