#!/bin/bash
# Which channel of the estimate costs the success? One channel at a time replaced by the ground truth (seeds 21-22), after the all-channel run has finished its 4 seeds.
cd /home/haktan/Projects/DroneRacing
while [ "$(grep -c '^2[1-4] ' runs/cam_eval/gt_ablate_g17.txt 2>/dev/null)" -lt 4 ]; do sleep 20; done
ps -eo pid,cmd | grep -E "run_oracle_isaac" | grep -v grep | awk '{print $1}' | xargs -r kill
sleep 2; docker ps -q --filter ancestor=lunarsim-isaaclab:6.0.1 | xargs -r docker stop >/dev/null; sleep 3
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
B="--gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ablate-gt"
for ch in gate vel up; do
  SEEDS="21 22" scripts/cam_eval.sh gt_only_$ch runs/g17 $B --ablate $ch --ekf-args "$KP" $SPEC
done
touch runs/cam_eval/ABLATE_DONE
exec scripts/run_oracle_isaac.sh
