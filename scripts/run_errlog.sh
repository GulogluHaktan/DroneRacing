#!/bin/bash
# usage: run_dagger.sh <estimator.pt> <tag> <seeds...>  -- closed-loop data (policy on estimator output)
cd /home/haktan/Projects/DroneRacing
est=$1; tag=$2; shift 2
for s in "$@"; do
  scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/t8snap --gatenet runs_host/g2/gatenet.pt --estimator $est --n-envs 24 --steps 3500 --seed $s --dump runs/flight/${tag}_$s 2>&1 | grep -E "VISION STACK|DUMP DONE|Traceback|Error" | head -4
done
echo DONE
