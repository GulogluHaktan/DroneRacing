#!/bin/bash
# Corrupt frames: drop the whole frame (--frame-check, 15 % of frames rejected in d14_cropgt) vs drop only the band (--band-check) vs no check.
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/DIAG14_DONE ]; do sleep 60; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g14/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST0="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4"
export NENV=4 SEEDS="21 22"
scripts/cam_eval.sh d14_band runs/g17 $CROP $ST0 --band-check
scripts/cam_eval.sh d14_nocheck runs/g17 $CROP $ST0
touch runs/cam_eval/DIAG14B_DONE
