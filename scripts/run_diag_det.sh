#!/bin/bash
cd /home/haktan/Projects/DroneRacing
until grep -q DONE runs/ekf_dump2.log; do sleep 20; done
: > runs/diag_det.log
scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/h100 --gatenet runs_host/g3/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 2500 --ekf --seed 51 --diag 2>&1 | grep -E "gate in analytic|dist |corner err|VISION STACK|Traceback|Error" | cut -c1-260 >> runs/diag_det.log
echo DONE >> runs/diag_det.log
