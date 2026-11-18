#!/bin/bash
# Why is perception in the loop low at 1-3 m (g14 crop line, seed 21)? Replace the filter crop window / filter roll by the truth.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g14/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --frame-check"
export NENV=4 SEEDS="21"
scripts/cam_eval.sh d14_cropgt runs/g17 $CROP $ST --crop-gt
scripts/cam_eval.sh d14_rollgt runs/g17 $CROP $ST --derotate-gt
scripts/cam_eval.sh d14_bothgt runs/g17 $CROP $ST --crop-gt --derotate-gt
touch runs/cam_eval/DIAG14_DONE
