#!/bin/bash
# Own-camera line with delay compensation (g17, GateNet g5, EKF5, 128 deg, 50 ms delay): 30 Hz (4 seeds) and 60 Hz (4 seeds, rate sensitivity).
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
GN="--gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-delay 5 --ekf5"
SEEDS="21 22 23 24" scripts/cam_eval.sh iso_g17_hzdelay_comp runs/g17 $GN --cam-hz 30 --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh own_g17_hz60_comp runs/g17 $GN --cam-hz 60 --ekf-args "$KP" $SPEC
touch runs/cam_eval/OWNCAM2_DONE
