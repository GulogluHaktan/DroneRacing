#!/bin/bash
# camera line with the real-size-gate policy g1 (saved args set gate_body): GateNet2 + EKF3 vs g3 + EKF; waits for the h100 queue to finish
cd /home/haktan/Projects/DroneRacing
while ! grep -q "^DONE" runs/cam_fixed.txt 2>/dev/null; do sleep 30; done
for seed in 21 22 41 51; do
  for v in g5ekf3 g3ekf; do
    if grep -q "^$v $seed " runs/cam_g1pol.txt 2>/dev/null; then continue; fi
    if [ $v = g5ekf3 ]; then ARGS="--gatenet runs_host/g5/gatenet.pt --ekf3"; else ARGS="--gatenet runs_host/g3/gatenet.pt"; fi
    out=$(scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g1 $ARGS --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf --seed $seed 2>&1 | grep -E "VISION STACK|estimator in|Traceback|Error|out of memory" | cut -c1-240 | tr '\n' ' ')
    echo "$v $seed $out" >> runs/cam_g1pol.txt
  done
done
echo DONE >> runs/cam_g1pol.txt
