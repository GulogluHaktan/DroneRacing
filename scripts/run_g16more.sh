#!/bin/bash
# g34 + g16 detector on 8 fresh seeds (29 30 35 36 41-44), disjoint from d11 training seeds 31-34 and from 21-28.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g16/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
NENV=4 SEEDS="29 30 35 36 41 42 43 44" scripts/cam_eval.sh t34g16b_cal runs/g34 $CROP $ST
touch runs/cam_eval/T34G16B_DONE
