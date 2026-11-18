#!/bin/bash
# g37 = g34 trained 1000 more iterations in the same estimator-in-the-loop env (perception penalty 2, own camera 30 Hz 50 ms).
# Reason: crash anatomy of g34+g17 on seeds 35/36 shows belief error 0.10 m but the policy believed a 0.8 m miss and flew the line anyway (policy side).
# Isaac target line with the g17 detector on the same 16 seeds as g16/g17 (21-30,35,36,41-44).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g37/model.pt ] && grep -q "it 1000" runs/g37.log 2>/dev/null || flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/g34 --ent-coef 0.001 --init-std 0.3 --iters 1000 --perc-pen 2 --ekf-loop --out runs/g37 > runs/g37.log 2>&1
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g17/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
NENV=4 SEEDS="21 22 23 24 25 26 27 28 29 30 35 36 41 42 43 44" scripts/cam_eval.sh t37_cal runs/g37 $CROP $ST
touch runs/cam_eval/T37_DONE
