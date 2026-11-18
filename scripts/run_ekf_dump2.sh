#!/bin/bash
cd /home/haktan/Projects/DroneRacing
for seed in 41 42 43; do
  scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/h100 --gatenet runs_host/g3/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3500 --ekf --seed $seed --dump runs/flight/h100r_$seed 2>&1 | grep -E "VISION STACK|estimator in|DUMP DONE|Traceback|Error" | cut -c1-250 >> runs/ekf_dump2.log
done
echo DONE >> runs/ekf_dump2.log
