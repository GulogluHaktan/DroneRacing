#!/bin/bash
# which part of the estimator output limits the camera line? replace vel / up / gate by the env oracle one at a time (h100, g3 + EKF); waits for the g1 queue
cd /home/haktan/Projects/DroneRacing
while ! grep -q "^DONE" runs/cam_g1pol.txt 2>/dev/null; do sleep 30; done
for seed in 21 22; do
  for ab in vel up gate "vel,up,gate"; do
    key="ab_${ab//,/+}"
    if grep -q "^$key $seed " runs/cam_ablate.txt 2>/dev/null; then continue; fi
    out=$(scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/h100 --gatenet runs_host/g3/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf --ablate "$ab" --seed $seed 2>&1 | grep -E "VISION STACK|Traceback|Error|out of memory" | cut -c1-240 | tr '\n' ' ')
    echo "$key $seed $out" >> runs/cam_ablate.txt
  done
done
echo DONE >> runs/cam_ablate.txt
