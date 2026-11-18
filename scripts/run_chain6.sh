#!/bin/bash
# Swift-style residual model + heavy-tail retrain (overnight). Order keeps one GPU job at a time.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
KAPPA='{"kappa":0.3,"kappa_corner":0.15}'
while [ ! -f runs/cam_eval/TAIL_DONE ]; do sleep 60; done
# 1) closed-loop error traces of the deployed camera line (seeds 41-46, disjoint from the eval seeds 21-32)
for s in 41 42 43 44 45 46; do
  [ -f runs/flight/sw_$s/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g6 --estimator runs_host/e4/estimator.pt \
    --n-envs 24 --steps 3000 --ekf --seed $s --gatenet runs_host/g5/gatenet.pt --ekf3 --ekf-args "$KAPPA" --dump runs/flight/sw_$s > runs/sw_$s.log 2>&1
done
$PY -I scripts/build_resid_bank.py runs/flight/sw_4[1-6]/data.pt --out runs_host/resid/bank_v1.pt > runs/bank_v1.log 2>&1
# 2) retrain from g6: g11 = calibrated heavy tail, g12 = residual bank
base="--profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/g6 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --ent-coef 0.001 --finish-time 150 --soft-frac 1.0 --soft-ramp 300 --v-max-start 30 --v-max 30 --iters 2500"
# g11 (tail 0.2/7) dropped: that tail is per-gate persistent; g6 falls to 18 % under it while the camera gives ~55 %
[ -f runs/g12/model.pt ] || $PY -I scripts/train_state.py $base --resid-bank runs_host/resid/bank_v1.pt --out runs/g12 > runs/g12.log 2>&1
# 3) torch cross-evaluation: each policy under parametric noise, calibrated tail and the residual bank
for r in g6 g12; do
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body --tail-prob 0 --resid-bank "" > runs/x_${r}_param.eval 2>&1
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body --tail-prob 0.2 --tail-mult 7 --resid-bank "" > runs/x_${r}_tail.eval 2>&1
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body --tail-prob 0 --resid-bank runs_host/resid/bank_v1.pt > runs/x_${r}_bank.eval 2>&1
done
# 4) camera line (old 130 deg, kappa 0.3), seeds 21-24, 24 drones
for r in g12; do
  SEEDS="21 22 23 24" scripts/cam_eval.sh ${r}_k30 runs/$r --gatenet runs_host/g5/gatenet.pt --ekf3 --ekf-args "$KAPPA"
done
SEEDS="24" scripts/cam_eval.sh kappa30 runs/g6 --gatenet runs_host/g5/gatenet.pt --ekf3 --ekf-args "$KAPPA"
touch runs/cam_eval/SWIFT_DONE
