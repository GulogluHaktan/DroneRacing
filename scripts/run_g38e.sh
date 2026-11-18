#!/bin/bash
# g38e = g38d continued 600 iterations (path-speed was 19.4 m/s in the surrogate eval, goal 20): --slow-pen 5, --slow-min 23, --crash-pen 40, --init-std 0.08. Goal: mean speed >= 20 m/s with success > 75 % (Isaac).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body $SN $SPEC"
flock runs/.isaac.lock $PY -I scripts/train_state.py $base --finish-time ${FT:-600} --gate-speed ${GS:-10} --slow-pen ${SP:-5} --slow-min ${SMIN:-23} --crash-pen ${CP:-40} --v-max-start ${VSTART:-30} --v-max 30 --init ${INIT:-runs/g38d} --ent-coef 0.001 --init-std ${ISTD:-0.08} --iters ${ITERS:-600} --perc-pen 2 --ekf-loop --out runs/g38e > runs/g38e.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g38e --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop > runs/surr_g38e.eval 2>&1
touch runs/G38E_DONE
