#!/bin/bash
# MonoRace perception on our camera: collect adaptive crops (d8), train GateNetMR, then the camera line with QuAdGate + GateEKF8 (eval_vision --mr).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 20; done
[ -f runs/percep/d8/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/collect_percep.py --task runs/g17 --frames 30000 --hfov 128 --img-h 308 --img-w 410 --crop-out 192 \
  --n-envs 24 --dist 1.0 18 --pitch-range 0 45 --roll-max 60 --yaw-noise 0.7 --lateral 2.5 --inverted-frac 0.10 --extr-deg 3 --seed 18 --preview 4 --out runs/percep/d8 > runs/d8_collect.log 2>&1
[ -f runs_host/m0/monorace.pt ] && [ -f runs_host/m0/DONE ] || { $PY -I scripts/train_monorace_net.py --data runs/percep/d8/data.pt --epochs 40 --out runs_host/m0 > runs/m0_train.log 2>&1; touch runs_host/m0/DONE; }
touch runs/cam_eval/MRTRAIN_DONE
