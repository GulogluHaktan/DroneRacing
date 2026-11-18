#!/bin/bash
cd /home/haktan/Projects/DroneRacing
while pgrep -f "scripts/run_eval_t10" > /dev/null; do sleep 15; done
echo "== dbg g3 t10"; scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/t10 --gatenet runs_host/g3/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf --seed 61 --dump runs/flight/dbg_g3 2>&1 | grep -E "VISION STACK|DUMP DONE|Traceback|Error"
echo "== dbg sim t10"; scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/t10 --gatenet runs_host/g3/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf --sim-det 1 --seed 61 --dump runs/flight/dbg_sim 2>&1 | grep -E "VISION STACK|DUMP DONE|Traceback|Error"
echo DONE
