#!/bin/bash
# after g6 is trained: camera line on seeds 21, 22 (g3 + EKF and GateNet2 + EKF3), then the estimator ablation for g1 (one Isaac container at a time)
cd /home/haktan/Projects/DroneRacing
while ! grep -q "g6 trained" runs/g6.txt 2>/dev/null; do sleep 30; done
for seed in 21 22; do
  for v in g3ekf g5ekf3; do
    if grep -q "^$v $seed " runs/cam_g6.txt 2>/dev/null; then continue; fi
    if [ $v = g5ekf3 ]; then ARGS="--gatenet runs_host/g5/gatenet.pt --ekf3"; else ARGS="--gatenet runs_host/g3/gatenet.pt"; fi
    out=$(scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g6 $ARGS --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf --seed $seed 2>&1 | grep -E "VISION STACK|estimator in|Traceback|out of memory" | cut -c1-240 | tr '\n' ' ')
    echo "$v $seed $out" >> runs/cam_g6.txt
  done
done
echo DONE >> runs/cam_g6.txt
for seed in 21 22; do
  for ab in vel up gate "vel,up,gate"; do
    key="ab_${ab//,/+}"
    if grep -q "^$key $seed " runs/cam_ablate.txt 2>/dev/null; then continue; fi
    out=$(scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g1 --gatenet runs_host/g3/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf --ablate "$ab" --seed $seed 2>&1 | grep -E "VISION STACK|Traceback|out of memory" | cut -c1-240 | tr '\n' ' ')
    echo "$key $seed $out" >> runs/cam_ablate.txt
  done
done
echo DONE >> runs/cam_ablate.txt
