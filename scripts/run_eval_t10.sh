#!/bin/bash
cd /home/haktan/Projects/DroneRacing
pol=${1:-runs/t10}
while pgrep -f "scripts/run_chain4" > /dev/null; do sleep 15; done
for g in g3 g2; do for seed in 21 22; do
  echo "== $pol EKF $g seed $seed"
  scripts/run_isaaclab.sh isaac/eval_vision.py --policy $pol --gatenet runs_host/$g/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 2000 --ekf --seed $seed 2>&1 | grep -E "VISION STACK|estimator in|Traceback|Error"
done; done
echo DONE
