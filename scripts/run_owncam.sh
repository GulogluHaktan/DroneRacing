#!/bin/bash
# Own-camera Isaac line (IMX219 estimate: 128 deg, 30 Hz, 45 deg tilt, latency) for g17 with GateNet g5 + EKF3 + aided_perm.
# A = clean (30 Hz + 50 ms delay only), B = + artifacts (rolling shutter 20 ms, 10 % corrupt frames, 3 deg mount error).
cd /home/haktan/Projects/DroneRacing
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 30; done   # user's ollama model must be unloaded
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
GN="--gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5"
SEEDS="21 22 23 24" scripts/cam_eval.sh own_g17_A runs/g17 $GN --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh own_g17_B runs/g17 $GN --rs-ms 20 --corrupt 0.1 --tilt-err 3 --ekf-args "$KP" $SPEC
touch runs/cam_eval/OWNCAM_DONE
