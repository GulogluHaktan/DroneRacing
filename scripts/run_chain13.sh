#!/bin/bash
# After the 2026-10-23 break: aided_perm is the pipeline now. (1) finish g15 + fix seeds 43-44, (2) g15p + fix 8 seeds,
# (3) g15b + fix 4 seeds. 2025 replica, camera line. Nothing else may run on the machine meanwhile (render breaks).
cd /home/haktan/Projects/DroneRacing
GN="--gatenet runs_host/g5/gatenet.pt --ekf3"
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
SEEDS="43 44" scripts/cam_eval.sh g15_perm runs/g15 $GN --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24 41 42 43 44" scripts/cam_eval.sh g15p_perm runs/g15p $GN --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh g15b_perm runs/g15b $GN --ekf-args "$KP" $SPEC
touch runs/cam_eval/CHAIN13_DONE
