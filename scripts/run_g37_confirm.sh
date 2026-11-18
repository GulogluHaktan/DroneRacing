#!/bin/bash
# Confirmation of g37 (g34 + 1000 iterations) with the g17 detector on 16 FRESH seeds (71-86): none of them was used for detector training
# (31-34, 51-54, 61-64) or for any earlier evaluation (21-30, 35, 36, 41-44). Same target line as t37_cal (flags unchanged).
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g17/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
NENV=4 SEEDS="71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86" scripts/cam_eval.sh t37c_cal runs/g37 $CROP $ST
touch runs/cam_eval/T37C_DONE
