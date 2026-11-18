#!/bin/bash
# Camera-line surrogate check: g17 and g27 in the torch env with the actor on GateEKF5 + simulated detections (--ekf-loop).
# Isaac target line: g17 18/32 = 56 %, g27 23/32 = 72 %. If the surrogate ranks them the same, it is a fast evaluator and a training env.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
for r in g17 g27 g32; do
  flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$r --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop > runs/surr_${r}.eval 2>&1
done
touch runs/SURR_DONE
