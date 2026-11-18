#!/bin/bash
# Geometry-aware gate passage: g33 = g27 (best, perception penalty 2) fine-tuned with the oriented body box (--gate-box) for 1500
# iterations. Half of the target-line crashes are oblique entries at gate 4 where the body grazes the frame. The Isaac evaluation
# reads gate_box from the saved args, so g33 is judged with the stricter box rule (g27: sphere, 23/32).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/G36_DONE ]; do sleep 60; done  # after the calibrated-surrogate policy g36
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --gate-box --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g33/model.pt ] && grep -q "it 1500" runs/g33.log 2>/dev/null || flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/g27 --ent-coef 0.001 --init-std 0.3 --iters 1500 --perc-pen 2 --out runs/g33 > runs/g33.log 2>&1
for r in g27 g33; do flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body --gate-box $SPEC $SN --resid-bank "" --profile own_drone > runs/box_${r}.eval 2>&1; done
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
NENV=4 SEEDS="21 22 23 24 25 26 27 28" scripts/cam_eval.sh t33_cal runs/g33 $CROP $ST
touch runs/cam_eval/T33_DONE
