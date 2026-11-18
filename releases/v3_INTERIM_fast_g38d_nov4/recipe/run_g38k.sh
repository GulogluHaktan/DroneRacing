#!/bin/bash
# g38k = g38d continued with a FOCUSED curriculum: 60 % of the mid-track starts use the crash gates 5 6 7 16 17 18 at 8-24 m/s (the old mid-track starts were 0-10 m/s), 40 % of all resets are mid-track starts (more starts from rest), launch grace 3 s (no sprint from the start). Evidence: Isaac crash anatomy of g38d: gate 6 at 22-28 m/s and the start at gate 0. Goal: mean speed >= 20 m/s with success > 75 % in Isaac (g38d: 21.1 m/s, 75.0 % over 96 flights).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body $SN $SPEC"
flock runs/.isaac.lock $PY -I scripts/train_state.py $base --finish-time ${FT:-600} --gate-speed ${GS:-10} --slow-pen ${SP:-4} --slow-min ${SMIN:-23} --crash-pen ${CP:-40} --slow-grace ${SG:-3.0} --mid-prob ${MP:-0.4} --mid-speed 8 24 --mid-focus 5 6 7 16 17 18 --mid-focus-prob ${MFP:-0.6} --v-max-start ${VSTART:-30} --v-max 30 --init ${INIT:-runs/g38d} --ent-coef 0.001 --init-std ${ISTD:-0.1} --iters ${ITERS:-900} --perc-pen 2 --ekf-loop --out runs/g38k > runs/g38k.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g38k --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop > runs/surr_g38k.eval 2>&1
touch runs/G38K_DONE
