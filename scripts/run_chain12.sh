#!/bin/bash
# Split-S fix: attitude-aided PnP tries 8 corner orders (rotations + mirror). Camera line on the 2025 replica, same seeds as the g15 baseline.
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/G15X_DONE ]; do sleep 60; done
GN="--gatenet runs_host/g5/gatenet.pt --ekf3"
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
SEEDS="21 22 23 24" scripts/cam_eval.sh g15_perm runs/g15 $GN --ekf-args "$KP" $SPEC
SEEDS="41 42 43 44" scripts/cam_eval.sh g15_perm runs/g15 $GN --ekf-args "$KP" $SPEC
touch runs/cam_eval/PERM_DONE
