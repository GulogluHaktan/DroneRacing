#!/bin/bash
# o1 and o2 against opponents on the NEW start layout (surrogate), plus crash anatomy of o2.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
: > runs/surr_opp2.txt
for cfg in "o1 1" "o2 0" "o2 1"; do
  set -- $cfg
  out=$(flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$1 --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop --opp-prob $2 2>&1)
  echo "$1 opp=$2 | $(echo "$out" | grep -E '^ *ALL' | head -1 | cut -c1-200)" >> runs/surr_opp2.txt
  echo "$out" | grep "crash causes" | head -1 >> runs/surr_opp2.txt
done
flock runs/.isaac.lock $PY -I scripts/opp_anatomy.py --run runs/o2 > runs/opp_anatomy_o2.txt 2>&1
touch runs/SURR_OPP2_DONE
