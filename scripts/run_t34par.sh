#!/bin/bash
# Parallel test: a second Isaac container (no lock) runs g34 seeds from the top of the list while run_t34more.sh runs from the bottom.
# Results go to t34p_cal.txt; cam_eval's blank-render guard rejects broken runs.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
for seed in 36 35 34; do
  grep -q "^$seed " runs/cam_eval/t34_cal.txt 2>/dev/null && continue
  mkdir -p runs/cam_eval/logs
  scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g34 --estimator runs_host/e4/estimator.pt --n-envs 4 --steps 3000 --ekf --seed $seed $CROP $ST 2>&1 \
    | grep -v "omniclient\|OmniHub" | tee runs/cam_eval/logs/t34p_cal_$seed.log | grep -E "VISION STACK|estimator in|gate obs err|PERCEPTION in loop" | cut -c1-400 | tr '\n' ' ' > /tmp/claude-1000/t34p_$seed.res
  res=$(cat /tmp/claude-1000/t34p_$seed.res)
  if echo "$res" | grep -qE "render blank ([5-9][0-9]|100)\."; then echo "$seed BLANK $res" >> runs/cam_eval/t34p_cal.txt; echo "blank renders in parallel: stop"; break; fi
  echo "$res" | grep -q "VISION STACK" && echo "$seed $res" >> runs/cam_eval/t34p_cal.txt
done
