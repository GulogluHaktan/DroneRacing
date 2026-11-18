#!/bin/bash
# Closed-loop test of the filter settings found in the 30 Hz + delay replay sweep (scripts/ekf5_sweep.py):
# chi2 9, sig_aided (0.07, 0.02), kappa 0.2 (replay held-out flights: gate error p90 0-5 m 0.342 -> 0.217 m).
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/OWNCAM2_DONE ]; do sleep 30; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KT='{"kappa":0.2,"kappa_corner":0.15,"aided_perm":true,"chi2":9.0,"sig_aided":[0.07,0.02]}'
GN="--gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-delay 5 --ekf5 --cam-hz 30"
SEEDS="21 22 23 24" scripts/cam_eval.sh own_g17_tuned30 runs/g17 $GN --ekf-args "$KT" $SPEC
touch runs/cam_eval/TUNED_DONE
