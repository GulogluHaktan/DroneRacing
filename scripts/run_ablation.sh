#!/bin/bash
# per-channel estimator ablation, sequential (parallel Isaac processes OOM the 8 GB GPU)
cd /home/haktan/Projects/DroneRacing
for seed in 21 22; do
  for ab in "" vel up gate vel,up,gate; do
    echo "== seed $seed ablate=[$ab]"
    scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/t8snap --gatenet runs_host/g2/gatenet.pt --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 2000 --seed $seed --ablate "$ab" 2>&1 | grep -E "VISION STACK|estimator in|Traceback"
  done
done
echo DONE
