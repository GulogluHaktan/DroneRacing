#!/bin/bash
# Standard camera-line evaluation (Phase 0 protocol): 4 seeds x 24 drones, GPU must be free of trainers, mean over seeds.
# usage: scripts/cam_eval.sh <tag> <policy dir> [extra eval_vision args...]   -> runs/cam_eval/<tag>.txt
cd /home/haktan/Projects/DroneRacing
tag=$1; pol=$2; shift 2
mkdir -p runs/cam_eval; out=runs/cam_eval/$tag.txt
NENV=${NENV:-24}; SEEDS=${SEEDS:-"21 22 23 24"}
for seed in $SEEDS; do
  if grep -q "^$seed " $out 2>/dev/null; then continue; fi
  while ps -eo cmd | grep -v grep | grep -q "train_state.py\|train_gatenet"; do echo "waiting: trainer on GPU"; sleep 60; done
  for try in 1 2 3 4; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt ${GPU_MAX:-1500} ]; do echo "waiting: GPU memory in use by another job"; sleep 30; done
    mkdir -p runs/cam_eval/logs
    res=$(flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy $pol --estimator runs_host/e4/estimator.pt --n-envs $NENV --steps 3000 --ekf --seed $seed "$@" 2>&1 \
          | grep -v "omniclient\|OmniHub" | tee runs/cam_eval/logs/${tag}_${seed}.log | grep -E "VISION STACK|estimator in|gate obs err|PERCEPTION in loop|WARMUP FAILED|Traceback|Error|out of memory" | cut -c1-400 | tr '\n' ' ')
    if echo "$res" | grep -qE "render blank [1-9][0-9]*\."; then echo "invalid run (blank renders), retry" >&2; res="BLANK $res"; continue; fi
    echo "$res" | grep -q "VISION STACK" && break
  done
  echo "$seed $res" >> $out
done
/home/haktan/isaac-env/bin/python -I - "$out" <<'PY'
import re, sys
s = [float(m.group(1)) for l in open(sys.argv[1]) if (m := re.search(r"success\s+([\d.]+)%", l))]
v = [float(m.group(1)) for l in open(sys.argv[1]) if (m := re.search(r"path-speed\s+([\d.]+)", l))]
if s:
    print(f"MEAN success {sum(s)/len(s):.1f}% over {len(s)} seeds (min {min(s):.1f}, max {max(s):.1f}); speed {sum(v)/max(1,len(v)):.1f} m/s")
PY
