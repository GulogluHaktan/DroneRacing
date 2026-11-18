#!/bin/bash
# g38b = g38a (iteration 300 of the plain "limit removed" run: speed stayed at 16.9 m/s, like the old v-max-30 run h100) continued with
# a direct reward for speed: --finish-time 300 (was 150) and --gate-speed 2 (reward at gate pass ~ speed/10, capped at 25 m/s).
# v-max curriculum continues from 21 (where g38a stopped) to 30 over the first 60 % of the iterations. Same env as g34/g37.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body $SN $SPEC"
flock runs/.isaac.lock $PY -I scripts/train_state.py $base --finish-time ${FT:-300} --gate-speed ${GS:-2} --v-max-start ${VSTART:-21} --v-max 30 --init ${INIT:-runs/g38a} --ent-coef 0.001 --init-std 0.3 --iters ${ITERS:-700} --perc-pen 2 --ekf-loop --out runs/g38b > runs/g38b.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g38b --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop > runs/surr_g38b.eval 2>&1
touch runs/G38B_DONE
