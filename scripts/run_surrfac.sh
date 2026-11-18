#!/bin/bash
# "factors" surrogate: original analytic detector (its error level matched Isaac for g17 / g27 / g32) x the measured relative view and
# body-rate dependence of the real detector. Validation against Isaac: g17 56.3, g27 71.9, g32 78.1, g34 78.1 %.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
for r in g32 g34 g17 g27; do flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$r --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop --ekf-loop-calib runs_host/detcal/g15_factors.pt > runs/surrfac_${r}.eval 2>&1; done
touch runs/SURRFAC_DONE
