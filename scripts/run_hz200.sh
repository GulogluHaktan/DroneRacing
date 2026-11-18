#!/bin/bash
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
COMMON="--profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/h100 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995"
$PY -I scripts/train_state.py $COMMON --out runs/h200 --policy-hz 200 --iters 2500 --ent-coef 0.001 --finish-time 150 --soft-frac 1.0 --soft-ramp 800 --v-max-start 14 --v-max 30 > runs/h200.log 2>&1
echo "== h200" >> runs/hz200.log
$PY -I scripts/eval_state.py --run runs/h200 --hall-prob 0.8 --drag-prob 0.25 --n-envs 2048 --max-iters 3000 2>&1 | grep ALL >> runs/hz200.log
echo DONE >> runs/hz200.log
