#!/bin/bash
# crop384 gave 0 % on 7 of 12 seeds (gates 7-24 %): diagnose seed 24 (detection rate, corner error, saved frames) after base8
cd /home/haktan/Projects/DroneRacing
while [ "$(grep -c . runs/cam_eval/base8.txt 2>/dev/null)" -lt 12 ]; do sleep 60; done
flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g6 --estimator runs_host/e4/estimator.pt --n-envs 8 --steps 3000 --ekf --seed 24 \
  --gatenet runs_host/g10/gatenet.pt --ekf3 --hfov 155 --img-h 616 --img-w 820 --crop 384 --ekf-args '{"sig_px_corner":6.0,"assoc_px_corner":60.0,"dup_px":12.0}' \
  --diag --save-frames runs/cropdiag > runs/cropdiag.log 2>&1
touch runs/cam_eval/CROPDIAG_DONE
