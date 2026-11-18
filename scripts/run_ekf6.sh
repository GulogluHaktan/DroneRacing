#!/bin/bash
# GateEKF6 (online camera mount estimate): with a 3 deg unknown mount error, without stress (no harm?), and under the combined stress. Seeds 21-24, 24 drones, GateNet g5.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
B="--gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf6 --aug-stable"
SEEDS="21 22 23 24" scripts/cam_eval.sh ekf6_tilt3 runs/g17 $B --tilt-err 3 --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh ekf6_clean runs/g17 $B --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh ekf6_stress_all runs/g17 $B --rs-ms 20 --corrupt 0.1 --tilt-err 3 --ekf-args "$KP" $SPEC
touch runs/cam_eval/EKF6_DONE
exec scripts/run_chain17b.sh
