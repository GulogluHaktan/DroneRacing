#!/bin/bash
# crop384 re-run after the PnP focal-length fix (det_features used the 130 deg default at 155 deg), then the artefact suite
cd /home/haktan/Projects/DroneRacing
while [ "$(grep -c . runs/cam_eval/base8.txt 2>/dev/null)" -lt 12 ]; do sleep 60; done
NENV=8 SEEDS="21 22 23 24 25 26 27 28 29 30 31 32" scripts/cam_eval.sh crop384 runs/g6 --gatenet runs_host/g10/gatenet.pt --ekf3 --hfov 155 --img-h 616 --img-w 820 --crop 384 \
  --ekf-args '{"sig_px_corner":6.0,"assoc_px_corner":60.0,"dup_px":12.0}'
touch runs/cam_eval/CROP2_DONE
rm -f runs/cam_eval/art_base.txt
scripts/run_artifacts.sh
