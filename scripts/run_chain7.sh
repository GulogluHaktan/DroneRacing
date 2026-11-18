#!/bin/bash
# 1) g12 camera evals (first run failed: bank path inside the Isaac container), A2RL 2025 replica first (user request).
# 2) "Safe model" g13 (user target >=70-80 % success first, speed later): g6 noise model (the residual bank did not help g12
#    in torch), progress capped at 13 m/s, 30 % of episodes on a randomized replica of the A2RL 2025 final track.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
KAPPA='{"kappa":0.3,"kappa_corner":0.15}'
REP="--fixed-prob 1 --fixed-jitter 1.5 0.2 --fixed-scale 0.85 1.15 --fixed-mirror 0.5"
GN="--gatenet runs_host/g5/gatenet.pt --ekf3"
SEEDS="21" scripts/cam_eval.sh g12_a2rl25x runs/g12 $GN --ekf-args "$KAPPA" --fixed-prob 1
SEEDS="21 22" scripts/cam_eval.sh g12_a2rl25 runs/g12 $GN --ekf-args "$KAPPA" $REP
SEEDS="21 22 23 24" scripts/cam_eval.sh g12_k30 runs/g12 $GN --ekf-args "$KAPPA"
touch runs/cam_eval/REP_G12_DONE
base="--profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/g6 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --ent-coef 0.001 --soft-frac 1.0 --soft-ramp 300 --iters 2500"
[ -f runs/g13/model.pt ] || $PY -I scripts/train_state.py $base --v-max-start 20 --v-max 13 --finish-time 75 \
  --fixed-prob 0.3 --fixed-jitter 1.5 0.2 --fixed-scale 0.85 1.15 --fixed-mirror 0.5 --out runs/g13 > runs/g13.log 2>&1
$PY -I scripts/eval_state.py --run runs/g13 --n-envs 4096 --gate-body --fixed-prob 0 > runs/x_g13_param.eval 2>&1
$PY -I scripts/eval_state.py --run runs/g13 --n-envs 4096 --gate-body --fixed-prob 0 --resid-bank runs_host/resid/bank_v1.pt > runs/x_g13_bank.eval 2>&1
$PY -I scripts/eval_state.py --run runs/g13 --n-envs 4096 --gate-body --fixed-prob 1 > runs/x_g13_a2rl25.eval 2>&1
$PY -I scripts/eval_state.py --run runs/g13 --n-envs 4096 --gate-body $REP > runs/x_g13_a2rl25dr.eval 2>&1
$PY -I scripts/eval_state.py --run runs/g13 --n-envs 4096 --gate-body $REP --resid-bank runs_host/resid/bank_v1.pt > runs/x_g13_a2rl25dr_bank.eval 2>&1
SEEDS="21" scripts/cam_eval.sh g13_a2rl25x runs/g13 $GN --ekf-args "$KAPPA" --fixed-prob 1
SEEDS="21 22" scripts/cam_eval.sh g13_a2rl25 runs/g13 $GN --ekf-args "$KAPPA" $REP
SEEDS="21 22 23 24" scripts/cam_eval.sh g13_k30 runs/g13 $GN --ekf-args "$KAPPA" --fixed-prob 0
touch runs/cam_eval/SAFE_DONE
