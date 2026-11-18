#!/bin/bash
# sequential queue (one GPU job at a time): finish crop384 -> diagnose crop seed 24 -> reward study
cd /home/haktan/Projects/DroneRacing
while pgrep -f "scripts/cam_eval.s[h] crop384" > /dev/null; do sleep 60; done
touch runs/cam_eval/CROP2_DONE
flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g6 --estimator runs_host/e4/estimator.pt --n-envs 8 --steps 3000 --ekf --seed 24 \
  --gatenet runs_host/g10/gatenet.pt --ekf3 --hfov 155 --img-h 616 --img-w 820 --crop 384 --ekf-args '{"sig_px_corner":6.0,"assoc_px_corner":60.0,"dup_px":12.0}' \
  --diag --save-frames runs/cropdiag > runs/cropdiag.log 2>&1
touch runs/cam_eval/CROPDIAG_DONE
scripts/run_reward.sh > runs/run_reward.out 2>&1
