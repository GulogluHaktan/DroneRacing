#!/bin/bash
# o1 = g37 (safe model) fine-tuned for racing against 1-3 scripted opponent drones (no camera/RL, gate-line followers, 6-13 m/s).
# Estimator-in-the-loop env as for g37; opponent features (nearest 2, camera FOV, distance-scaled noise) appended to the observation,
# new input columns start at zero (train_state --init handles it). Contact with an opponent = crash; proximity penalty within 1 m.
# 70 % of the episodes have opponents (30 % solo keep the solo skill). Compare solo success afterwards with scripts/eval_state.py.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/g37 --ent-coef 0.001 --init-std 0.3 --iters ${ITERS:-800} --perc-pen 2 --ekf-loop \
  --opp-max 3 --opp-prob 0.7 --opp-speed 6 13 --opp-near 0.5 --out runs/o1 > runs/o1.log 2>&1
touch runs/O1_DONE
