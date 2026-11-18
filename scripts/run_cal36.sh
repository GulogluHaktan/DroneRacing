#!/bin/bash
# Calibrated surrogate: the analytic detector of the estimator-in-the-loop env is fitted to the real detector in the Isaac loop
# (d11 percep logs while g34 flies). g34 exploited the uncalibrated detector (surrogate 91 %, Isaac ~75 %).
# 1. fit; 2. surrogate g32 / g34 (Isaac: 78.1 % / ~75 %); 3. g36 = g32 trained in the calibrated loop with perception DR;
# 4. surrogate normal + bad day; 5. Isaac target line 8 seeds.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/percep/d11/plog_34.pt ]; do sleep 60; done
CAL=runs_host/detcal/g15.pt
[ -f $CAL ] || $PY -I scripts/fit_detector.py runs/percep/d11/plog_3[1-4].pt --usable --out $CAL > runs/fit_detector.log 2>&1  # usable = whole-gate detection or >= 2 corner peaks (logs with the corner column)
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
for r in g32 g34; do flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$r --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop --ekf-loop-calib $CAL > runs/surrcal_${r}.eval 2>&1; done
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 50 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g36/model.pt ] && grep -q "it 1000" runs/g36.log 2>/dev/null || flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/g32 --ent-coef 0.001 --init-std 0.3 --iters 1000 --perc-pen 2 --ekf-loop --ekf-loop-dr --ekf-loop-calib $CAL --out runs/g36 > runs/g36.log 2>&1
flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/g36 --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop --ekf-loop-calib $CAL > runs/surrcal_g36.eval 2>&1
for r in g32 g36; do flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$r --n-envs 2048 --gate-body $SPEC $SN --resid-bank "" --profile own_drone --ekf-loop --ekf-loop-calib $CAL --ekf-loop-dr bad_day > runs/surrcalbad_${r}.eval 2>&1; done
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
NENV=4 SEEDS="21 22 23 24 25 26 27 28" scripts/cam_eval.sh t36_cal runs/g36 $CROP $ST
touch runs/G36_DONE
