#!/bin/bash
# g38 = FAST model: g37 (safe model) trained with the speed limit removed from the reward: --v-max 30 (cap on progress per step, was 13),
# --v-max-start 14 (curriculum, linear to 30 over the first 60 % of the iterations), --finish-time 150 (bonus for a fast finish, was 75).
# Everything else as g34/g37: estimator-in-the-loop env (--ekf-loop, own camera 30 Hz, 50 ms), perception penalty 2, own_drone profile.
# Checkpoint every 50 iterations (runs/g38/model.pt). To continue after an interruption: --init runs/g38 --v-max-start 30.
# Then the surrogate eval (overrates trained-in policies: confirm in Isaac with scripts/run_g38_isaac.sh).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --finish-time 150 $SN $SPEC"
INIT=${INIT:-runs/g37}; VSTART=${VSTART:-14}
flock runs/.isaac.lock $PY -I scripts/train_state.py $base --v-max-start $VSTART --v-max 30 --init $INIT --ent-coef 0.001 --init-std 0.3 --iters ${ITERS:-1100} --perc-pen 2 --ekf-loop --out runs/g38 > runs/g38.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g38 --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop > runs/surr_g38.eval 2>&1
touch runs/G38_DONE
