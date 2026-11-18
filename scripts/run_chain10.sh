#!/bin/bash
# g16 (user request): g15 with the progress speed cap raised 13 -> 30 m/s (everything else identical); how fast does it get?
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
KAPPA='{"kappa":0.3,"kappa_corner":0.15}'
GN="--gatenet runs_host/g5/gatenet.pt --ekf3"
SN="--split-noise 3 2 1.5 1.5"
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
while [ ! -f runs/cam_eval/G15_DONE ]; do sleep 60; done
base="--profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/g15 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --ent-coef 0.001 --iters 1200 --v-max-start 13 --v-max 30 --finish-time 75"
[ -f runs/g16/model.pt ] || $PY -I scripts/train_state.py $base $SN $SPEC --out runs/g16 > runs/g16.log 2>&1
$PY -I scripts/eval_state.py --run runs/g16 --n-envs 4096 --gate-body $SPEC $SN > runs/x_g16_spec_sn.eval 2>&1
$PY -I scripts/eval_state.py --run runs/g16 --n-envs 4096 --gate-body $SPEC --sensor-scale 0 > runs/x_g16_spec_oracle.eval 2>&1
$PY -I scripts/eval_state.py --run runs/g15 --n-envs 4096 --gate-body $SPEC --sensor-scale 0 > runs/x_g15_spec_oracle.eval 2>&1
SEEDS="21 22 23 24" scripts/cam_eval.sh g16_a2rl25x runs/g16 $GN --ekf-args "$KAPPA" $SPEC
touch runs/cam_eval/G16_DONE
