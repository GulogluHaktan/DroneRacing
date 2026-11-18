#!/bin/bash
# Isaac closed loop with the new perception (GateNet g14: realcam training, 820x616 full-resolution 384 adaptive crop, roll derotation),
# own camera (128 deg, 30 Hz, 50 ms delay compensated), policy g17, 2025 replica. Perception in the loop: "PERCEPTION in loop" lines.
# 1. clean and stressed lines (rolling shutter 20 ms, 10 % corrupt frames, 40 % dark frames, corrupt-frame check), 4 seeds x 8 drones;
#    reference: g13 full frame (410x308) under the same stress.
# 2. target test: + unknown 3 deg mount error per drone; calibration flight (seed 41) -> offline calibration -> race seeds 21-24 without / with it.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while ps -eo cmd | grep -v grep | grep -q "train_gatenet2.py"; do sleep 30; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g14/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
FULL13="--gatenet runs_host/g13/gatenet.pt --ekf3 --hfov 128 --img-h 308 --img-w 410 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --frame-check"
export SEEDS="21 22 23 24 25 26 27 28"  # 820x616 render: 4 drones per seed (8 gave 100 % blank renders)
NENV=4 scripts/cam_eval.sh i_g14_stress runs/g17 $CROP $ST
SEEDS="21 22 23 24" NENV=4 scripts/cam_eval.sh i_g14_clean runs/g17 $CROP
SEEDS="21 22 23 24" NENV=12 scripts/cam_eval.sh i_g13_stress runs/g17 $FULL13 $ST
touch runs/cam_eval/IG14_DONE
# target test
MT="--tilt-err 3 --tilt-seed 7"
mkdir -p runs_host/calib
[ -f runs/flight/cal14_41/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g17 --estimator runs_host/e4/estimator.pt \
  --n-envs 4 --steps 3000 --ekf --seed 41 $CROP $ST $MT --dump runs/flight/cal14_41 > runs/cal14_41_dump.log 2>&1
[ -f runs_host/calib/cal14_41.pt ] || $PY -I scripts/calib_extrinsic.py runs/flight/cal14_41/data.pt --drones 4 --steps 1500 --kappa 0 --h 616 --w 820 --out runs_host/calib/cal14_41.pt > runs/cal14_41_calib.log 2>&1
NENV=4 scripts/cam_eval.sh t14_uncal runs/g17 $CROP $ST $MT
NENV=4 scripts/cam_eval.sh t14_cal runs/g17 $CROP $ST $MT --extr-corr runs_host/calib/cal14_41.pt
touch runs/cam_eval/T14_DONE
