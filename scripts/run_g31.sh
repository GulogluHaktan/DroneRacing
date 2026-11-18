#!/bin/bash
# g31 = g27 trained 1500 more iterations with the same perception penalty (2; penalty 4 in g30 gave no gain). Target line seeds 21-28.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g31/model.pt ] && grep -q "it 1500" runs/g31.log 2>/dev/null || flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/g27 --ent-coef 0.001 --init-std 0.25 --iters 1500 --perc-pen 2 --out runs/g31 > runs/g31.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g31 --n-envs 4096 --gate-body $SPEC $SN --resid-bank "" --profile own_drone > runs/g31_param.eval 2>&1
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
NENV=4 SEEDS="21 22 23 24 25 26 27 28" scripts/cam_eval.sh t31_cal runs/g31 $CROP $ST
touch runs/cam_eval/T31_DONE
