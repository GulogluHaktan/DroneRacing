#!/bin/bash
# o2 = o1 trained 1000 more iterations with the realistic start layout (opponents beside / behind with >= 1 m lateral gap, or >= 3.5 m ahead / behind on our line); 80 % of the episodes with opponents.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/o1 --ent-coef 0.001 --init-std 0.3 --iters ${ITERS:-1000} --perc-pen 2 --ekf-loop \
  --opp-max 3 --opp-prob 0.8 --opp-speed 6 13 --opp-near 0.5 --out runs/o2 > runs/o2.log 2>&1
touch runs/O2_DONE
