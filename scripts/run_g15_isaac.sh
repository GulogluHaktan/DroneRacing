#!/bin/bash
# Goal: perception >= 90 % and flight >= 80 % under real-camera stress. GateNet g15 (g14 + close-range data d10), 820x616 384 crop,
# roll derotation, band reject of corrupt rows; g17; own camera 30 Hz, 50 ms delay compensated; 4 drones per seed (render limit).
# 1. stress line seeds 21-28 (compare with g14 band reject: seeds 21 / 22 = 86.3 / 85.1 % perception, 50 / 75 % flight).
# 2. target: + unknown 3 deg mount error per drone, calibration flight (seed 41) -> offline calibration -> race seeds 21-28 with it.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/CLOSE15_DONE ]; do sleep 60; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20"
export NENV=4
SEEDS="21 22 23 24 25 26 27 28" scripts/cam_eval.sh g15_stress runs/g17 $CROP $ST
touch runs/cam_eval/G15S_DONE
MT="--tilt-err 3 --tilt-seed 7"
mkdir -p runs_host/calib
[ -f runs/flight/cal15_41/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g17 --estimator runs_host/e4/estimator.pt \
  --n-envs 4 --steps 3000 --ekf --seed 41 $CROP $ST $MT --dump runs/flight/cal15_41 > runs/cal15_41_dump.log 2>&1
[ -f runs_host/calib/cal15_41.pt ] || $PY -I scripts/calib_extrinsic.py runs/flight/cal15_41/data.pt --drones 4 --steps 1500 --kappa 0 --h 616 --w 820 --out runs_host/calib/cal15_41.pt > runs/cal15_41_calib.log 2>&1
SEEDS="21 22 23 24 25 26 27 28" scripts/cam_eval.sh t15_cal runs/g17 $CROP $ST $MT --extr-corr runs_host/calib/cal15_41.pt
touch runs/cam_eval/T15_DONE
