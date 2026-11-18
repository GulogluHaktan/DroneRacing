#!/bin/bash
# Mount error study (seeds 21-22, 24 drones, g17, GateNet g5, n_map 32, stable appearance, 30 Hz, 50 ms):
# (a) EKF6 with a faster estimate (gain 0.1, gates down to 12 px) under 3 deg; (b) EKF5 with 1 and 2 deg (how well must the mount be calibrated).
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
KF='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32,"ext_gain":0.1,"ext_min_ext":12.0}'
B="--gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --aug-stable"
SEEDS="21 22" scripts/cam_eval.sh tilt1_ekf5 runs/g17 $B --ekf5 --tilt-err 1 --ekf-args "$KP" $SPEC
SEEDS="21 22" scripts/cam_eval.sh tilt2_ekf5 runs/g17 $B --ekf5 --tilt-err 2 --ekf-args "$KP" $SPEC
SEEDS="21 22" scripts/cam_eval.sh tilt3_ekf6fast runs/g17 $B --ekf6 --tilt-err 3 --ekf-args "$KF" $SPEC
touch runs/cam_eval/TILTSTUDY_DONE
