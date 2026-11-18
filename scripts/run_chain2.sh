#!/bin/bash
# after the reward study: re-run the crop384 seeds that collapsed to 0 % (24, 27) plus 21, now with the render-health counters
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/REWARD_DONE ]; do sleep 60; done
NENV=8 SEEDS="24 27 21" scripts/cam_eval.sh crop384_rh runs/g6 --gatenet runs_host/g10/gatenet.pt --ekf3 --hfov 155 --img-h 616 --img-w 820 --crop 384 \
  --ekf-args '{"sig_px_corner":6.0,"assoc_px_corner":60.0,"dup_px":12.0}'
touch runs/cam_eval/RH_DONE
