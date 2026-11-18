#!/bin/bash
# Value of the 820x616 @ 85 Hz driver mode (architecture L0; KAIST uses this IMX219 mode): g27 on the full target line with the camera at
# 85 Hz (exact frame times) and the rolling-shutter readout of that mode (11.6 ms, compensated). Compare with t27_cal (30 Hz): 23/32.
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/T30_DONE ]; do sleep 60; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 85 --cam-hz-exact --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 11.6 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 11.6 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
NENV=4 SEEDS="21 22 23 24 25 26 27 28" scripts/cam_eval.sh t27_85hz runs/g27 $CROP $ST
touch runs/cam_eval/T27_85_DONE
