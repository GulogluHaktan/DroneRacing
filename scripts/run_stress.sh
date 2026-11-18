#!/bin/bash
# How close is the 80 % line to a real camera? Same line (g17, GateNet g5, EKF5, n_map 32, stable appearance, 30 Hz, 50 ms delay), one realism stress at a time, seeds 21-24.
cd /home/haktan/Projects/DroneRacing
while [ "$(grep -c '^4[1-4] ' runs/cam_eval/own_g17_stable30_b.txt 2>/dev/null)" -lt 4 ]; do sleep 30; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
B="--gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP"
SEEDS="21 22 23 24" scripts/cam_eval.sh stress_rs_corrupt_tilt runs/g17 --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --rs-ms 20 --corrupt 0.1 --tilt-err 3 --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh stress_delay_mismatch runs/g17 --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf-delay 3 --ekf5 --aug-stable --ekf-args "$KP" $SPEC
touch runs/cam_eval/STRESS_DONE
exec scripts/run_chain17b.sh
