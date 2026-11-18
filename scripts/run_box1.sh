#!/bin/bash
# Geometry-aware gate passage: oriented body box (attitude at the crossing) in training and evaluation. g25 = g17 fine-tuned with --gate-box.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --gate-box --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g25/model.pt ] || $PY -I scripts/train_state.py $base --init runs/g17 --ent-coef 0.001 --init-std 0.3 --iters 1500 --out runs/g25 > runs/g25.log 2>&1
$PY -I scripts/eval_state.py --run runs/g25 --n-envs 4096 --gate-body --gate-box $SPEC $SN --resid-bank "" --profile own_drone > runs/own_g25_box.eval 2>&1
$PY -I scripts/eval_state.py --run runs/g25 --n-envs 4096 --gate-body $SPEC $SN --resid-bank "" --profile own_drone > runs/own_g25_sphere.eval 2>&1
touch runs/BOX1_DONE
