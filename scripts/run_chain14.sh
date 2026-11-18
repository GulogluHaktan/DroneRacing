#!/bin/bash
# MonoRace-style perception: (1) collect 130 deg 256x192 frames (deployed camera line) with more rolled / inverted views,
# (2) train GateSeg, (3) camera line on the 2025 replica: g15p (or g15) + GateSeg + aided_perm vs the GateNet result.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/cam_eval/CHAIN13_DONE ]; do sleep 60; done
[ -f runs/percep/d6/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/collect_percep.py --task runs/h100 --frames 24000 --hfov 130 --img-h 192 --img-w 256 \
  --n-envs 64 --dist 1.2 16 --pitch-range 0 45 --roll-max 60 --yaw-noise 0.7 --lateral 2.5 --inverted-frac 0.10 --seed 11 --out runs/percep/d6 > runs/d6_collect.log 2>&1
[ -f runs_host/s0/gateseg.pt ] || $PY -I scripts/train_gateseg.py --data runs/percep/d6/data.pt --epochs 12 --out runs_host/s0 > runs/s0_train.log 2>&1
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
SEEDS="21 22 23 24" scripts/cam_eval.sh g15p_seg runs/g15p --gatenet runs_host/g5/gatenet.pt --ekf3 --gateseg runs_host/s0/gateseg.pt --ekf-args "$KP" $SPEC
touch runs/cam_eval/SEG_DONE
