#!/bin/bash
# Surrogate with the adaptive-crop rule (only gates inside the window around the next gate can be detected). Isaac: g32 78.1 = g34 78.1.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/SURRBAD_DONE ]; do sleep 20; done
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
for r in g32 g34 g27; do flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$r --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop --ekf-loop-calib runs_host/detcal/g15_factors_crop.pt > runs/surrcrop_${r}.eval 2>&1; done
touch runs/SURRCROP_DONE
