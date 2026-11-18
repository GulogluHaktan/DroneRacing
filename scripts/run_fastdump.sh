#!/bin/bash
# Diagnostic dumps of the FAST policy g38d with the detector g19 on fresh seeds 95 96 (same flags as the target-line eval, with --dump).
# Question: where does the velocity error of the filter grow with the speed (IMU scale error, gaps in the detections, delay)?
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
for s in 95 96; do
  [ -f runs/flight/fast_$s/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g38d --estimator runs_host/e4/estimator.pt \
    --n-envs 4 --steps 3000 --ekf --seed $s --gatenet runs_host/g19/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args "$KP" $SPEC $ST \
    --dump runs/flight/fast_$s > runs/fastdump_$s.log 2>&1
done
touch runs/FASTDUMP_DONE
