#!/bin/bash
# g15: track specialist (user idea, as MonoRace/Swift): only the A2RL 2025 layout, small jitter (map-level 0.3 m / 3 deg, no scale/mirror).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
KAPPA='{"kappa":0.3,"kappa_corner":0.15}'
GN="--gatenet runs_host/g5/gatenet.pt --ekf3"
SN="--split-noise 3 2 1.5 1.5"
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
while [ ! -f runs/g14/model.pt ]; do sleep 60; done
base="--profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/g13 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --ent-coef 0.001 --iters 1200 --v-max-start 13 --v-max 13 --finish-time 75"
[ -f runs/g15/model.pt ] || $PY -I scripts/train_state.py $base $SN $SPEC --out runs/g15 > runs/g15.log 2>&1
for r in g13 g15; do
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body $SPEC $SN > runs/x_${r}_spec_sn.eval 2>&1
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body --fixed-prob 1 --fixed-jitter 1.5 0.2 --fixed-scale 0.85 1.15 --fixed-mirror 0.5 $SN > runs/x_${r}_dr_sn.eval 2>&1
done
SEEDS="21 22 23 24" scripts/cam_eval.sh g15_a2rl25x runs/g15 $GN --ekf-args "$KAPPA" $SPEC
touch runs/cam_eval/G15_DONE
