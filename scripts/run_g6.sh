#!/bin/bash
# g6: g1 recipe with milder sensing errors (sensor scale 0.6): is the torch noise profile too pessimistic for the camera line?
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
$PY -I scripts/train_state.py --profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/h100 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --ent-coef 0.001 --finish-time 150 --soft-frac 1.0 --soft-ramp 800 --v-max-start 14 --v-max 30 --iters 2500 --out runs/g6 > runs/g6.log 2>&1
echo "== g6 trained" >> runs/g6.txt
