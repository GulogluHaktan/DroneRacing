#!/bin/bash
# g17 = g16 fine-tuned with extra on-policy crops collected while g34 flies with the g16 detector (seeds 51-54).
# Eval seeds stay disjoint from all training seeds (31-34, 51-54): same 16 seeds as g16 (21-30,35,36,41-44) for a direct comparison.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
mkdir -p runs/percep/d12
for s in 51 52 53 54; do
  [ -f runs/percep/d12/data_$s.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g34 --estimator runs_host/e4/estimator.pt \
    --n-envs 4 --steps 3000 --ekf --seed $s --gatenet runs_host/g16/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args "$KP" $SPEC $ST \
    --save-crops runs/percep/d12 --percep-log runs/percep/d12/plog_$s.pt > runs/d12_collect_$s.log 2>&1
done
[ -f runs_host/g17/gatenet.pt ] || flock runs/.isaac.lock $PY -I scripts/train_gatenet2.py --data runs/percep/d9/data.pt runs/percep/d10/data.pt runs/percep/d11/data_3[1-4].pt runs/percep/d12/data_5[1-4].pt \
  --max-per 10000 --init runs_host/g16/gatenet.pt --epochs 4 --bs 16 --lr 1e-4 --realcam --derotate 3 --holdout --out runs_host/g17 > runs/g17_train.log 2>&1
CROP="--gatenet runs_host/g17/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
NENV=4 SEEDS="21 22 23 24 25 26 27 28 29 30 35 36 41 42 43 44" scripts/cam_eval.sh t34g17_cal runs/g34 $CROP $ST
touch runs/cam_eval/T34G17_DONE
