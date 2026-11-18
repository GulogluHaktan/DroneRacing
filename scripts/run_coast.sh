#!/bin/bash
# Can the filter coast on the IMU through the last metres before a gate? Detections dropped when the estimated gate distance < D.
# Same line and stress as i_g14_stress (seeds 21-23); compare success with i_g14_stress (detections used down to 0 m).
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/DIAG14B_DONE ]; do sleep 60; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g14/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check"
export NENV=4 SEEDS="21 22"
scripts/cam_eval.sh c14b_coast25 runs/g17 $CROP $ST --min-det-dist 2.5
scripts/cam_eval.sh c14b_coast15 runs/g17 $CROP $ST --min-det-dist 1.5
touch runs/cam_eval/COAST_DONE
