#!/bin/bash
# g38f = g38d continued with a strong gate-centre bonus (--center-rew 8, was 1): the crash anatomy of g38d + g19 (seeds 95, 96) shows frame hits where the filter error is small (0.06-0.35 m) and the policy believed a ~1 m miss (gate 6 at 22-28 m/s, gate 0 at the start). Goal: path-speed >= 20 m/s with success > 75 % (Isaac).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body $SN $SPEC"
flock runs/.isaac.lock $PY -I scripts/train_state.py $base --finish-time ${FT:-600} --gate-speed ${GS:-10} --slow-pen ${SP:-4} --slow-min ${SMIN:-23} --crash-pen ${CP:-40} --center-rew ${CR:-8} --v-max-start ${VSTART:-30} --v-max 30 --init ${INIT:-runs/g38d} --ent-coef 0.001 --init-std ${ISTD:-0.1} --iters ${ITERS:-800} --perc-pen 2 --ekf-loop --out runs/g38f > runs/g38f.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g38f --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop > runs/surr_g38f.eval 2>&1
touch runs/G38F_DONE
