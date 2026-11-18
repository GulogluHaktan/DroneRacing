#!/bin/bash
# camera-line comparison on h100: GateNet2 + EKF3 vs g3 + EKF (one Isaac container at a time; needs free VRAM)
cd /home/haktan/Projects/DroneRacing
for seed in 41 42 43 51; do
  for v in g5ekf3 g3ekf; do
    if grep -q "^$v $seed " runs/cam_g5.txt 2>/dev/null; then continue; fi
    if [ $v = g5ekf3 ]; then ARGS="--gatenet runs_host/g5/gatenet.pt --ekf3"; else ARGS="--gatenet runs_host/g3/gatenet.pt"; fi
    out=$(scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/h100 $ARGS --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf --seed $seed 2>&1 | grep -E "VISION STACK|estimator in|Traceback|Error|out of memory" | cut -c1-240 | tr '\n' ' ')
    echo "$v $seed $out" >> runs/cam_g5.txt
  done
done
echo DONE >> runs/cam_g5.txt
