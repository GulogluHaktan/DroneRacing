#!/bin/bash
# g34 = g32 fine-tuned estimator-in-the-loop (--ekf-loop: actor sees GateEKF5 fed by simulated detections, own camera 30 Hz 50 ms),
# perception penalty 2. The surrogate matches Isaac (g17 55.0 vs 56.3 %, g27 68.6 vs 71.9 %). Surrogate eval, then Isaac target line.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g34/model.pt ] && grep -q "it 1000" runs/g34.log 2>/dev/null || flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/g32 --ent-coef 0.001 --init-std 0.3 --iters 1000 --perc-pen 2 --ekf-loop --out runs/g34 > runs/g34.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g34 --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop > runs/surr_g34.eval 2>&1
touch runs/G34_SURR_DONE
