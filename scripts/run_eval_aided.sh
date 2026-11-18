#!/bin/bash
cd /home/haktan/Projects/DroneRacing
for spec in "runs/t10 g3 21" "runs/t10 g3 22" "runs/t8snap g3 21" "runs/t10 g2 21" "runs/t10 g3 23"; do
  set -- $spec
  echo "== $1 aided-EKF $2 seed $3"
  scripts/run_isaaclab.sh isaac/eval_vision.py --policy $1 --gatenet runs_host/$2/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 2000 --ekf --seed $3 2>&1 | grep -E "VISION STACK|estimator in|Traceback|Error"
done
echo DONE
