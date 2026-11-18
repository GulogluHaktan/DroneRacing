#!/bin/bash
# Surrogate eval of the weight interpolations of g37 (safe) and g38d (fast): soupNN = (1-NN/100) * g37 + NN/100 * g38d.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
: > runs/surr_soup.txt
for pol in g37 soup50 soup70 soup85 g38d; do
  line=$(flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$pol --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop 2>&1 | grep -E "^ *ALL" | head -1 | cut -c1-200)
  echo "$pol | $line" >> runs/surr_soup.txt
done
touch runs/SURRSOUP_DONE
