#!/bin/bash
# Surrogate eval of o1 solo (opp-prob 0) and against 1-3 scripted opponents (opp-prob 1); g37 solo for reference.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
: > runs/surr_opp.txt
for cfg in "g37 0" "o1 0" "o1 1"; do
  set -- $cfg
  out=$(flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$1 --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop $( [ $1 = o1 ] && echo --opp-prob $2 ) 2>&1)
  echo "$1 opp=$2 | $(echo "$out" | grep -E '^ *ALL' | head -1 | cut -c1-200)" >> runs/surr_opp.txt
  echo "$out" | grep "crash causes" | head -1 >> runs/surr_opp.txt
done
touch runs/SURR_OPP_DONE
