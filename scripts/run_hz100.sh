#!/bin/bash
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
COMMON="--profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/t19 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995"
$PY -I scripts/train_state.py $COMMON --out runs/h100 --policy-hz 100 --iters 3000 --ent-coef 0.001 --finish-time 150 --soft-frac 1.0 --soft-ramp 1000 --v-max-start 14 --v-max 30 > runs/h100.log 2>&1
echo "== h100" >> runs/hz100.log
$PY -I scripts/eval_state.py --run runs/h100 --hall-prob 0.8 --drag-prob 0.25 --n-envs 2048 --max-iters 3000 2>&1 | grep ALL >> runs/hz100.log
echo DONE >> runs/hz100.log
