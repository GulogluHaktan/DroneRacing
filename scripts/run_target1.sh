#!/bin/bash
# Target test (goal 2026-10-26: flight >= 80 % with everything combined): g17 + GateNet g13, own camera 30 Hz, 50 ms delay compensated,
# rolling shutter 20 ms, 10 % corrupt frames, 40 % dark frames, corrupt-frame check, unknown 3 deg mount error per drone.
# 1. calibration flight (seed 41, same drones = --tilt-seed 7) with dump; 2. offline mount calibration (no ground truth used);
# 3. race flights seeds 21-24 without and with the calibrated correction.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/cam_eval/PERCEP1_DONE ]; do sleep 60; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
LINE="--gatenet runs_host/g13/gatenet.pt --ekf3 --hfov 128 --img-h 308 --img-w 410 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC --rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --frame-check --tilt-err 3 --tilt-seed 7"
mkdir -p runs_host/calib
[ -f runs/flight/cal_41/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g17 --estimator runs_host/e4/estimator.pt \
  --n-envs 12 --steps 3000 --ekf --seed 41 $LINE --dump runs/flight/cal_41 > runs/cal_41_dump.log 2>&1
[ -f runs_host/calib/cal_41.pt ] || $PY -I scripts/calib_extrinsic.py runs/flight/cal_41/data.pt --drones 12 --steps 1500 --kappa 0 --out runs_host/calib/cal_41.pt > runs/cal_41_calib.log 2>&1
export NENV=12 SEEDS="21 22 23 24"
scripts/cam_eval.sh t1_uncal runs/g17 $LINE
scripts/cam_eval.sh t1_cal runs/g17 $LINE --extr-corr runs_host/calib/cal_41.pt
touch runs/cam_eval/TARGET1_DONE
