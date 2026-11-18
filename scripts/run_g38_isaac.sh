#!/bin/bash
# Isaac target-line evaluation of a fast-model candidate (default g38d) with the g17 detector, seeds 71-78 (g37 on the same seeds: 30/32).
# Read success, lap and path-speed (compare with MonoRace: ~21 m/s mean, 16.56 s best lap, 28.2 m/s peak) per seed in runs/cam_eval/<tag>.txt.
# usage: [POL=runs/g38d TAG=t38d_cal SEEDS="71 72 ..."] scripts/run_g38_isaac.sh
cd /home/haktan/Projects/DroneRacing
POL=${POL:-runs/g38d}; TAG=${TAG:-t38d_cal}
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g17/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
NENV=4 SEEDS="${SEEDS:-71 72 73 74 75 76 77 78}" scripts/cam_eval.sh $TAG $POL $CROP $ST
touch runs/cam_eval/${TAG}_DONE
