#!/bin/bash
# heavy-tail check: measured closed-loop gate error p90 is 3-4x the torch model (tail_prob 0.2 / mult 7 reproduces it).
# torch: g6 vs g8 (trained with tail 0.15/4) under the calibrated tail; then g8 on the camera line (old 130 deg, seeds 21-23)
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/cam_eval/KAPPA_DONE ]; do sleep 60; done
for r in g6 g8; do
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body --tail-prob 0.2 --tail-mult 7 > runs/tail_$r.eval 2>&1
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body --tail-prob 0 > runs/notail_$r.eval 2>&1
done
SEEDS="21 22 23" scripts/cam_eval.sh g8_cam runs/g8 --gatenet runs_host/g5/gatenet.pt --ekf3
touch runs/cam_eval/TAIL_DONE
