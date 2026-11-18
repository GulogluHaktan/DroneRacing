#!/bin/bash
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
COMMON="--profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/h100 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --corner-vis --ent-coef 0.001 --finish-time 150 --soft-frac 1.0 --soft-ramp 800 --v-max-start 14 --v-max 30 --iters 2500"
run() { name=$1; shift; if grep -q "^== $name$" runs/age.log 2>/dev/null; then return; fi
  $PY -I scripts/train_state.py $COMMON --out runs/$name "$@" > runs/$name.log 2>&1
  echo "== $name" >> runs/age.log
  $PY -I scripts/eval_state.py --run runs/$name --hall-prob 0.8 --drag-prob 0.25 --n-envs 2048 --max-iters 6000 --gate-body --corner-vis 2>&1 | grep ALL | sed 's/^/   hall mix : /' >> runs/age.log
  $PY -I scripts/eval_state.py --run runs/$name --hall-prob 0.0 --drag-prob 0.0 --n-envs 2048 --max-iters 6000 --gate-body --corner-vis 2>&1 | grep ALL | sed 's/^/   old tracks: /' >> runs/age.log; }
run g4 --age-obs
run g5
echo DONE >> runs/age.log
