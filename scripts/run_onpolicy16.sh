#!/bin/bash
# On-policy detector data: g34 flies a different line than the data the detector was trained on (perception g34 87-89 % vs g32 ~90 %).
# 1. d11: crops the detector sees while g34 flies the target line (seeds 31-34, disjoint from the evaluation seeds), clean renders + labels.
# 2. g16 = g15 fine-tuned on d9 + d10 + d11 (realcam, derotation). 3. Isaac target line g34 + g16, seeds 21-28.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/G36_DONE ]; do sleep 60; done  # after the calibrated-surrogate policy (flight first)
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
mkdir -p runs/percep/d11
for s in 31 32 33 34; do
  [ -f runs/percep/d11/data_$s.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g34 --estimator runs_host/e4/estimator.pt \
    --n-envs 4 --steps 3000 --ekf --seed $s --gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args "$KP" $SPEC $ST \
    --save-crops runs/percep/d11 --percep-log runs/percep/d11/plog_$s.pt > runs/d11_collect_$s.log 2>&1
done
[ -f runs_host/g16/gatenet.pt ] || flock runs/.isaac.lock $PY -I scripts/train_gatenet2.py --data runs/percep/d9/data.pt runs/percep/d10/data.pt runs/percep/d11/data_3[1-4].pt \
  --max-per 10000 --init runs_host/g15/gatenet.pt --epochs 4 --bs 16 --lr 2e-4 --realcam --derotate 3 --holdout --out runs_host/g16 > runs/g16_train.log 2>&1
CROP="--gatenet runs_host/g16/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
NENV=4 SEEDS="21 22 23 24 25 26 27 28" scripts/cam_eval.sh t34g16_cal runs/g34 $CROP $ST
touch runs/cam_eval/T34G16_DONE
