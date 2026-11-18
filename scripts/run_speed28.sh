#!/bin/bash
# Safe model: g28 = g17 fine-tuned with a speed penalty above 14 m/s (--speed-pen 2). Torch check vs g17, then the full target line
# (g15 perception, stress, 3 deg mount error + calibration cal15_41) seeds 21-24; compare with t15_cal (g17).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/cam_eval/PERC27_DONE ]; do sleep 60; done
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g28/model.pt ] && grep -q "it 1000" runs/g28.log 2>/dev/null || flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/g17 --ent-coef 0.001 --init-std 0.3 --iters 1000 --speed-pen 2 --speed-cap 14 --out runs/g28 > runs/g28.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g28 --n-envs 4096 --gate-body $SPEC $SN --resid-bank "" --profile own_drone > runs/speed_g28_param.eval 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g28 --n-envs 4096 --gate-body $SPEC --resid-bank runs_host/resid/bank_own30.pt --resid-prob 1 --profile own_drone > runs/speed_g28_bank.eval 2>&1
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
NENV=4 SEEDS="21 22 23 24" scripts/cam_eval.sh t28_cal runs/g28 $CROP $ST
touch runs/cam_eval/SPEED28_DONE
