#!/bin/bash
# Goal 2026-10-26: perception >= 90 % (gates with >= 2 corners found) under real-camera effects, then flight >= 80 %.
# GPU order: g13 training -> d9 data (own camera 820x616, MonoRace 384 adaptive crops) -> g14 training on crops -> offline
# perception breakdowns -> policy sensitivity sweep -> Isaac perception tests -> target test.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while ps -eo cmd | grep -v grep | grep -q "train_gatenet2.py\|calib_extrinsic.py"; do sleep 30; done
[ -f runs/percep/d9/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/collect_percep.py --task runs/g17 --frames 24000 --hfov 128 --img-h 616 --img-w 820 \
  --crop-out 384 --extr-deg 3 --n-envs 8 --dist 1.0 18 --pitch-range 0 45 --roll-max 60 --yaw-noise 0.7 --lateral 2.5 --inverted-frac 0.10 --seed 23 --preview 4 --out runs/percep/d9 > runs/d9_collect.log 2>&1
[ -f runs_host/g14/gatenet.pt ] || $PY -I scripts/train_gatenet2.py --data runs/percep/d9/data.pt --init runs_host/g13/gatenet.pt --epochs 10 --bs 16 --realcam --derotate 3 --out runs_host/g14 > runs/g14_train.log 2>&1
{ echo "== g13 full frame d7"; $PY -I scripts/percep_breakdown.py runs_host/g13/gatenet.pt --n 1500
  echo "== g14 crop d9"; $PY -I scripts/percep_breakdown.py runs_host/g14/gatenet.pt --data runs/percep/d9/data.pt --n 1500 --derotate 3; } > runs/percep_breakdown.txt 2>&1
for s in 1.0 0.75 0.5 0.35 0.25 0.0; do
  echo "scale $s"; $PY -I scripts/eval_state.py --run runs/g17 --n-envs 4096 --gate-body --fixed-prob 1 --fixed-jitter 0.3 0.05 --resid-bank runs_host/resid/bank_own30.pt --resid-prob 1 --resid-scale $s --profile own_drone 2>&1 | grep -E " ALL |rror"
done > runs/sweep_bank_scale.txt
scripts/run_percep1.sh
# crop line (820x616 render, 8 drones per seed), same stress as run_percep1.sh
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
NENV=8 SEEDS="21 22 23 24" scripts/cam_eval.sh p2_g14crop_stress_fc runs/g17 --gatenet runs_host/g14/gatenet.pt --ekf3 --crop 384 --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args "$KP" $SPEC --rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --frame-check --derotate
touch runs/cam_eval/PERCEP2_DONE
exec scripts/run_target1.sh
