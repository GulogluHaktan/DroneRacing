#!/bin/bash
# Seed 23 fails with g15 (perception 90.8 %, flight 25 %): closed-loop dump + videos for the crash anatomy.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20"
[ -f runs/flight/g15_23/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g17 --estimator runs_host/e4/estimator.pt \
  --n-envs 4 --steps 3000 --ekf --seed 23 $CROP $ST --dump runs/flight/g15_23 --video runs/video_g15 > runs/g15_23_dump.log 2>&1
/home/haktan/isaac-env/bin/python -I scripts/crash_anatomy.py runs/flight/g15_23/data.pt > runs/g15_23_anatomy.txt 2>&1
touch runs/CRASH23_DONE
