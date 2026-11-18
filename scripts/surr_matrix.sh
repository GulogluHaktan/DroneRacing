#!/bin/bash
# Surrogate eval of the fast policies at different sensing/actuation severities (0.6 = training severity of g38d/g38f; 1.0 = g38g).
# Calibration idea: Isaac gave g38d 75.0 % (surrogate at 0.6: 84.1 %): which severity makes the surrogate match Isaac?
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
: > runs/surr_matrix.txt
for pol in g38d g38f g38g; do for sc in 0.6 0.8 1.0; do
  line=$(flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$pol --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop --sensor-scale $sc 2>&1 | grep -E "^ *ALL" | head -1 | cut -c1-200)
  echo "$pol scale $sc | $line" >> runs/surr_matrix.txt
done; done
touch runs/SURRMATRIX_DONE
