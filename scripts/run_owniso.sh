#!/bin/bash
# Isolate what breaks the own-camera line for g17 (seed 21 only, 24 drones): profile/tilt, 30 Hz gating, delay (ignored / compensated).
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
GN="--gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128"
run() { tag=$1; shift; NENV=${NENV:-24} SEEDS="21" scripts/cam_eval.sh $tag "$@" --ekf-args "$KP" $SPEC; }
run iso_g17_hz runs/g17 $GN --cam-hz 30
run iso_g17_hzdelay runs/g17 $GN --cam-hz 30 --cam-delay 5
run iso_g17_hzdelay_comp runs/g17 $GN --cam-hz 30 --cam-delay 5 --ekf5
touch runs/cam_eval/ISO_DONE
