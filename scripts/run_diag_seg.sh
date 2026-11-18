#!/bin/bash
# Detection quality in the closed loop (--diag): GateSeg at 410x308 against GateNet g5 at 256x192, same policy and filter, seed 21.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 20; done
flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g17 --estimator runs_host/e4/estimator.pt --n-envs 12 --steps 1500 --ekf --seed 21 --gatenet runs_host/g5/gatenet.pt --gateseg runs_host/s1/gateseg.pt --ekf3 --hfov 128 --img-h 308 --img-w 410 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --diag --ekf-args "$KP" $SPEC > runs/diag_seg.log 2>&1
flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g17 --estimator runs_host/e4/estimator.pt --n-envs 12 --steps 1500 --ekf --seed 21 --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --diag --ekf-args "$KP" $SPEC > runs/diag_gn.log 2>&1
touch runs/cam_eval/DIAGSEG_DONE
