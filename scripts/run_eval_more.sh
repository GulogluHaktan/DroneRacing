#!/bin/bash
cd /home/haktan/Projects/DroneRacing
for seed in 24 25 26 27; do
  echo "== runs/t10 aided-EKF g3 seed $seed"
  scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/t10 --gatenet runs_host/g3/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 2000 --ekf --seed $seed 2>&1 | grep -E "VISION STACK|estimator in|Traceback|Error"
done
echo DONE
