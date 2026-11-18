#!/bin/bash
# Confirmation of the 80 % result on four more seeds (41-44), then the detector chain.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
SEEDS="41 42 43 44" scripts/cam_eval.sh own_g17_stable30_b runs/g17 --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args "$KP" $SPEC
touch runs/cam_eval/CONFIRM_DONE
exec scripts/run_chain17b.sh
