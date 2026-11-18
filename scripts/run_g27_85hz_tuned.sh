#!/bin/bash
# 85 Hz camera with the filter scaled to the measurement rate (frame errors are correlated over consecutive frames):
# gate-correction gains x 30/85, corner / PnP / aided measurement noise x sqrt(85/30) = 1.68. First 85 Hz run with the 30 Hz tuning:
# velocity error 0.43 / 0.79 m/s (30 Hz: 0.29-0.42).
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.106,"kappa_corner":0.053,"aided_perm":true,"n_map":32,"sig_px_corner":5.0,"sig_pnp":[0.42,0.17],"sig_aided":[0.25,0.076]}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 85 --cam-hz-exact --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 11.6 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 11.6 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
NENV=4 SEEDS="21 22 23 24 25 26 27 28" scripts/cam_eval.sh t27_85hz_tuned runs/g27 $CROP $ST
touch runs/cam_eval/T27_85T_DONE
