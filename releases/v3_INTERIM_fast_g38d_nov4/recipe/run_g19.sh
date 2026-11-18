#!/bin/bash
# g19 = detector g17 fine-tuned on crops flown by the FAST policy g38d (19-20 m/s: more blur / rolling-shutter shift than the 15 m/s flights that made g17).
# Isaac showed g38d: detection at >= 1 m 89.0 % (g37 92.7 %), velocity error +26 %, near-gate error +30 % on the same seeds 71-77.
# 1. d14: crops while g38d flies the target line with g17, seeds 91-94 (disjoint from the evaluation seeds 71-78 and from all earlier training seeds).
# 2. g19 = g17 + d9 + d10 + d11 + d12 + d13 + d14 (lr 1e-4, 4 epochs).  3. Isaac eval of g38d + g19 on seeds 71-78 (tag t38d19_cal).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
mkdir -p runs/percep/d14
for s in 91 92 93 94; do
  [ -f runs/percep/d14/data_$s.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g38d --estimator runs_host/e4/estimator.pt \
    --n-envs 4 --steps 3000 --ekf --seed $s --gatenet runs_host/g17/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args "$KP" $SPEC $ST \
    --save-crops runs/percep/d14 --percep-log runs/percep/d14/plog_$s.pt > runs/d14_collect_$s.log 2>&1
done
[ -f runs_host/g19/gatenet.pt ] || flock runs/.isaac.lock $PY -I scripts/train_gatenet2.py --data runs/percep/d9/data.pt runs/percep/d10/data.pt runs/percep/d11/data_3[1-4].pt runs/percep/d12/data_5[1-4].pt runs/percep/d13/data_6[1-4].pt runs/percep/d14/data_9[1-4].pt \
  --max-per 10000 --init runs_host/g17/gatenet.pt --epochs 4 --bs 16 --lr 1e-4 --realcam --derotate 3 --holdout --out runs_host/g19 > runs/g19_train.log 2>&1
CROP="--gatenet runs_host/g19/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
NENV=4 SEEDS="71 72 73 74 75 76 77 78" scripts/cam_eval.sh t38d19_cal runs/g38d $CROP $ST
touch runs/cam_eval/T38D19_DONE
