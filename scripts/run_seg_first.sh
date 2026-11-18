#!/bin/bash
# MonoRace mask first (user priority 2026-10-25 22:00): train GateSeg on the own-camera frames d7, then the camera line (stable appearance, n_map 32, 30 Hz, 50 ms EKF5)
# on the 8 seeds and under the combined realism stress; then the rest of chain 17b (tuned filter, single stresses, GateNet2, YOLO).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 20; done
[ -f runs_host/s1/gateseg.pt ] || $PY -I scripts/train_gateseg.py --data runs/percep/d7/data.pt --epochs 10 --rs-ms-max 30 --out runs_host/s1 > runs/s1_train.log 2>&1
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
L="--gatenet runs_host/g5/gatenet.pt --gateseg runs_host/s1/gateseg.pt --ekf3 --hfov 128 --img-h 308 --img-w 410 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable"
NENV=12 SEEDS="21 22 23 24 41 42 43 44" scripts/cam_eval.sh seg_stable30 runs/g17 $L --ekf-args "$KP" $SPEC
NENV=12 SEEDS="21 22 23 24" scripts/cam_eval.sh seg_stress_all runs/g17 $L --rs-ms 20 --corrupt 0.1 --tilt-err 3 --ekf-args "$KP" $SPEC
touch runs/cam_eval/SEGFIRST_DONE
exec scripts/run_chain17b.sh
