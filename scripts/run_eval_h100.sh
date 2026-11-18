#!/bin/bash
# waits until >=6500 MiB of GPU memory is free (llama-server stopped), then runs the camera eval of h100
cd /home/haktan/Projects/DroneRacing
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 6500 ]; do sleep 20; done
: > runs/eval_h100.log
for seed in 21 22 23 24; do
  scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/h100 --gatenet runs_host/g3/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf --seed $seed 2>&1 | grep -E "VISION STACK|estimator in|Traceback|Error" | cut -c1-300 >> runs/eval_h100.log
done
echo DONE >> runs/eval_h100.log
