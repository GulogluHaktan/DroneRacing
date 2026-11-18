#!/bin/bash
# noise-free ground-truth ablation on g6 (GateNet2+EKF3) + g2 (perception penalty) on the camera line; waits for g7 camera queue
cd /home/haktan/Projects/DroneRacing
# camera evals are invalid while a trainer shares the GPU (render warm-up times out -> blank frames): wait for g8 training to finish
true
R="--gatenet runs_host/g5/gatenet.pt --ekf3 --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf"
for seed in 21 22; do
  for ab in vel up gate "vel,up,gate"; do
    key="gt_${ab//,/+}"
    if grep -q "^$key $seed " runs/cam_gt.txt 2>/dev/null; then continue; fi
    out=$(scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g6 $R --ablate "$ab" --ablate-gt --seed $seed 2>&1 | grep -E "VISION STACK|Traceback|Error|out of memory" | cut -c1-240 | tr '\n' ' ')
    echo "$key $seed $out" >> runs/cam_gt.txt
  done
  out=$(scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g2 $R --seed $seed 2>&1 | grep -E "VISION STACK|Traceback|Error|out of memory" | cut -c1-240 | tr '\n' ' ')
  echo "g2 $seed $out" >> runs/cam_gt.txt
done
out=$(scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g7 $R --seed 24 2>&1 | grep -E "VISION STACK|Traceback|Error|out of memory" | cut -c1-240 | tr '\n' ' ')
echo "g7rerun 24 $out" >> runs/cam_gt.txt
echo DONE >> runs/cam_gt.txt
