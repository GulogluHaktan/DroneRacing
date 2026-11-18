#!/bin/bash
# Rolling-shutter compensation of the corner measurements (gyro x row time): g14 crop line with band reject, seeds 21-22,
# compare with d14_band (same settings without the compensation).
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g14/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
NENV=4 SEEDS="21 22" scripts/cam_eval.sh r14_band_rscomp runs/g17 $CROP --rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20
touch runs/cam_eval/RSCOMP_DONE
