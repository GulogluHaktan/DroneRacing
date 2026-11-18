#!/bin/bash
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
$PY -I scripts/train_state.py --profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --no-obstacles --sensor-scale 0 --obs-history 8 --n-obs-gates 3 --split-prob 0.15 --gamma 0.995 --policy-hz 100 --ent-coef 0.001 --finish-time 150 --v-max-start 14 --v-max 30 --iters 600 --out runs/sanity1 > runs/sanity1.log 2>&1
$PY -I scripts/eval_state.py --run runs/sanity1 --hall-prob 0.8 --drag-prob 0.25 --n-envs 2048 --max-iters 6000 2>&1 | grep ALL > runs/sanity1_eval.log
