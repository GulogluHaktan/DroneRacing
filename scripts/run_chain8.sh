#!/bin/bash
# g14: g13 + split-S perception degradation in training (calibrated: g13 torch replica 49.8 % ~ camera 45-50 %).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
KAPPA='{"kappa":0.3,"kappa_corner":0.15}'
REP="--fixed-prob 1 --fixed-jitter 1.5 0.2 --fixed-scale 0.85 1.15 --fixed-mirror 0.5"
GN="--gatenet runs_host/g5/gatenet.pt --ekf3"
SN="--split-noise 3 2 1.5 1.5"
base="--profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/g13 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --ent-coef 0.001 --iters 2000 --v-max-start 13 --v-max 13 --finish-time 75"
[ -f runs/g14/model.pt ] || $PY -I scripts/train_state.py $base $SN --fixed-prob 0.3 --fixed-jitter 1.5 0.2 --fixed-scale 0.85 1.15 --fixed-mirror 0.5 --out runs/g14 > runs/g14.log 2>&1
for r in g13 g14; do
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body --fixed-prob 1 $SN > runs/x_${r}_a2rl25_sn.eval 2>&1
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body $REP $SN > runs/x_${r}_a2rl25dr_sn.eval 2>&1
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body --fixed-prob 0 > runs/x_${r}_param2.eval 2>&1
done
SEEDS="21 23" scripts/cam_eval.sh g14_a2rl25x runs/g14 $GN --ekf-args "$KAPPA" --fixed-prob 1
SEEDS="21 22" scripts/cam_eval.sh g14_a2rl25 runs/g14 $GN --ekf-args "$KAPPA" $REP
SEEDS="23" scripts/cam_eval.sh g13_a2rl25x runs/g13 $GN --ekf-args "$KAPPA" --fixed-prob 1
SEEDS="21 22 23 24" scripts/cam_eval.sh g14_k30 runs/g14 $GN --ekf-args "$KAPPA" --fixed-prob 0
touch runs/cam_eval/G14_DONE
