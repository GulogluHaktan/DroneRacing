#!/bin/bash
# t10 + g3 + aided EKF closed-loop dumps (error statistics for ekf-profile calibration + crash analysis)
cd /home/haktan/Projects/DroneRacing
for seed in 61 62; do
  echo "== dump t10 EKF g3 seed $seed"
  scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/t10 --gatenet runs_host/g3/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3500 --ekf --seed $seed --dump runs/flight/t10ekf_$seed 2>&1 | grep -E "VISION STACK|estimator in|DUMP DONE|Traceback|Error"
done
echo DONE
