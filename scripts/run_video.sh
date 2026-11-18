#!/bin/bash
# Clips of the own-camera line (g17, 128 deg, 30 Hz, 50 ms delay, EKF5): finished and crashed episodes, chase view + onboard view with true (green) and believed (cyan) gate.
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/OWNCAM3_DONE ]; do sleep 30; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 30; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
rm -rf runs/video_own; mkdir -p runs/video_own
flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g17 --estimator runs_host/e4/estimator.pt --n-envs 12 --steps 3000 --ekf --seed 21 \
  --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --ekf-args "$KP" $SPEC --video runs/video_own --video-envs 12 --video-max 2 > runs/video_own.log 2>&1
for d in runs/video_own/*/; do
  n=$(basename $d)
  ffmpeg -y -loglevel error -framerate 50 -i $d/%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 23 runs/video_own/$n.mp4
done
touch runs/cam_eval/VIDEO_DONE
