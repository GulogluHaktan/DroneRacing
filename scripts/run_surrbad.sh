#!/bin/bash
# Does the "bad day" surrogate (detector worse, delay mismatch 30-70 ms, 0.7 deg mount residual, 15 % lost frames) rank g32 = g34 like Isaac?
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/SURRFAC_DONE ]; do sleep 20; done
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
for r in g32 g34 g27 g17; do flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$r --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop --ekf-loop-calib runs_host/detcal/g15_factors.pt --ekf-loop-dr bad_day > runs/surrfacbad_${r}.eval 2>&1; done
touch runs/SURRBAD_DONE
