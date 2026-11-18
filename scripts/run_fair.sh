#!/bin/bash
# Fair comparison of the perception lines under the SAME stress and seeds as i_g14_stress (seeds 21-24):
# old GateNet g5 (410x308 full frame, clean training) and the new g13; plus g14 on the clean line.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
FULL="--ekf3 --hfov 128 --img-h 308 --img-w 410 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
CROP="--gatenet runs_host/g14/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --frame-check"
export SEEDS="21 22 23 24"
NENV=12 scripts/cam_eval.sh f_g5_stress runs/g17 --gatenet runs_host/g5/gatenet.pt $FULL $ST
NENV=4 scripts/cam_eval.sh i_g14_clean runs/g17 $CROP
NENV=12 scripts/cam_eval.sh f_g13_stress runs/g17 --gatenet runs_host/g13/gatenet.pt $FULL $ST
touch runs/cam_eval/FAIR_DONE
