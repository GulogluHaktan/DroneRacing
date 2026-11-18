#!/bin/bash
# Robustness margin for the real drone: perception domain randomization in the estimator loop (droneracing/ekf_loop.py DR_TRAIN:
# detector recall 0.85-0.97, noise x1-1.6, true delay 30-70 ms vs 50 ms assumed, mount residual up to 0.7 deg, up to 15 % lost frames).
# Surrogate on the normal line and on a fixed "bad day" (recall 0.85, noise x1.5, delay mismatch, 0.7 deg, 15 % lost).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/cam_eval/T34_DONE ]; do sleep 60; done  # after the Isaac confirmation of g34
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
EV="--n-envs 2048 --gate-body $SPEC $SN --resid-bank \"\" --profile own_drone --ekf-loop"
for r in g32 g34; do flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$r --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop --ekf-loop-dr bad_day > runs/surrbad_${r}.eval 2>&1; done
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g35/model.pt ] && grep -q "it 1000" runs/g35.log 2>/dev/null || flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/g34 --ent-coef 0.001 --init-std 0.3 --iters 1000 --perc-pen 2 --ekf-loop --ekf-loop-dr --out runs/g35 > runs/g35.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g35 --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop > runs/surr_g35.eval 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g35 --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop --ekf-loop-dr bad_day > runs/surrbad_g35.eval 2>&1
touch runs/G35_DONE
