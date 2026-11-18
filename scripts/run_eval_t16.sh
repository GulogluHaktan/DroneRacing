#!/bin/bash
cd /home/haktan/Projects/DroneRacing
for seed in 21 22 23 24 25 26 27; do
  scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/t16 --gatenet runs_host/g3/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3500 --ekf --seed $seed --dump runs/flight/t16_$seed 2>&1 | grep -E "VISION STACK|estimator in|Traceback|Error"
done
echo DONE
