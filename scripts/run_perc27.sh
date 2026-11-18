#!/bin/bash
# Flight: gate 4 has 2-3x the belief error of the other gates at 1-4 m (p50 0.15, p90 0.29 m) and a steep entry.
# (1) ceiling check: g17 + g15 perception on the clean line (no stress), seeds 21-24.
# (2) g27 = g17 + MonoRace perception penalty (--perc-pen 2: keep the next gate within 60 deg of the camera axis), g15 stress line seeds 21-24.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/cam_eval/T15_DONE ]; do sleep 60; done
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20"
NENV=4 SEEDS="21 22 23 24" scripts/cam_eval.sh g15_clean runs/g17 $CROP
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g27/model.pt ] && grep -q "it 1000" runs/g27.log 2>/dev/null || flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/g17 --ent-coef 0.001 --init-std 0.3 --iters 1000 --perc-pen 2 --out runs/g27 > runs/g27.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g27 --n-envs 4096 --gate-body $SPEC $SN --resid-bank "" --profile own_drone > runs/perc_g27_param.eval 2>&1
NENV=4 SEEDS="21 22 23 24" scripts/cam_eval.sh g27_g15_stress runs/g27 $CROP $ST
touch runs/cam_eval/PERC27_DONE
