#!/bin/bash
cd /home/haktan/Projects/DroneRacing
until grep -q "g7 trained" runs/g7.txt 2>/dev/null; do sleep 60; done
for seed in 21 22 23 24; do
  for v in g3ekf g5ekf3; do
    if [ $v = g5ekf3 ]; then ARGS="--gatenet runs_host/g5/gatenet.pt --ekf3"; else ARGS="--gatenet runs_host/g3/gatenet.pt"; fi
    out=$(scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g7 $ARGS --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf --seed $seed 2>&1 | grep -E "VISION STACK|estimator in|Traceback|Error|out of memory" | cut -c1-240 | tr '\n' ' ')
    echo "$v $seed $out" >> runs/cam_g7.txt
  done
done
echo DONE >> runs/cam_g7.txt
