#!/bin/bash
# Phase 1b: A2RL camera (155 deg, 616x820) + MonoRace adaptive 384 crop. 8 GB GPU: render budget ~4 Mpx -> 8 drones at 616x820.
cd /home/haktan/Projects/DroneRacing
if [ ! -f runs/percep/d5/data.pt ]; then
  flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/collect_percep.py --task runs/h100 --frames 20000 --hfov 155 --img-h 616 --img-w 820 --crop-out 384 --extr-deg 3 \
    --n-envs 8 --dist 1.0 18 --pitch-range 0 40 --roll-max 35 --yaw-noise 0.7 --lateral 2.5 --inverted-frac 0.03 --seed 11 --preview 4 --out runs/percep/d5 > runs/crop_collect.log 2>&1
fi
if [ ! -f runs_host/g10/gatenet.pt ]; then
  /home/haktan/isaac-env/bin/python -I scripts/train_gatenet2.py --data runs/percep/d5/data.pt --init runs_host/g5/gatenet.pt --epochs 10 --bs 16 --out runs_host/g10 > runs/crop_train.log 2>&1
fi
NENV=8 SEEDS="21 22 23 24 25 26 27 28 29 30 31 32" scripts/cam_eval.sh crop384 runs/g6 --gatenet runs_host/g10/gatenet.pt --ekf3 --hfov 155 --img-h 616 --img-w 820 --crop 384 \
  --ekf-args '{"sig_px_corner":6.0,"assoc_px_corner":60.0,"dup_px":12.0}'
touch runs/cam_eval/CROP_DONE
