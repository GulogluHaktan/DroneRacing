#!/bin/bash
# Isaac eval of o3 (camera pipeline of the final line) with rendered scripted opponents (opp-prob 1) and solo (opp-prob 0).
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g17/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
for cfg in "1 73" "1 74" "1 75" "1 76" "1 77" "1 78"; do
  set -- $cfg
  flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/o3 --estimator runs_host/e4/estimator.pt --n-envs 4 --steps 6000 --ekf --seed $2 \
    $CROP $ST --opp-prob 1 --opp-head runs_host/g17o2/opphead.pt --opp-hits 3 --opp-thr 0.5 > runs/isaac_o3camB_s$2.log 2>&1
done
touch runs/ISAAC_O3CAMB_DONE
