#!/bin/bash
# GateNet vs analytic detector vs ideal corners, same Isaac physics + estimator e4 + policy t8snap
cd /home/haktan/Projects/DroneRacing
for sd in 0 1; do
  for seed in 21 22; do
    echo "== sim-det $sd seed $seed"
    scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/t8snap --gatenet runs_host/g2/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 48 --steps 2000 --seed $seed --sim-det $sd 2>&1 | grep -E "VISION STACK|estimator in|Traceback"
  done
done
echo DONE
