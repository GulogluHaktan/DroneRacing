#!/bin/bash
# Factorial control: g17 at 128 deg with 100 Hz detections and no delay on seeds 21-24 (seed 21 is already in the file).
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/OWN2_DONE ]; do sleep 60; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
SEEDS="21 22 23 24" scripts/cam_eval.sh own_g17_fov100 runs/g17 --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --ekf-args "$KP" $SPEC
touch runs/cam_eval/OWNCAM3_DONE
