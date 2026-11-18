#!/bin/bash
# Sensitivity to the accuracy of the race map (approximate gate positions): map error 0.10 m / 0.05 m (default 0.25 m, yaw 0.09 rad), seeds 21-24, 24 drones.
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/ORACLE_DONE ]; do sleep 60; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
B="--gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable"
SEEDS="21 22 23 24" scripts/cam_eval.sh map010 runs/g17 $B --map-err 0.10 --map-yaw 0.04 --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh map005 runs/g17 $B --map-err 0.05 --map-yaw 0.02 --ekf-args "$KP" $SPEC
touch runs/cam_eval/MAPERR_DONE
exec scripts/run_chain17b.sh
