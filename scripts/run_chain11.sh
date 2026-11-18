#!/bin/bash
# Push the safe model g15 past 70 % on the 2025 replica: (A) polish (low std, no entropy bonus), (B) Swift-style bank from g15's
# own camera flights on this one track, mixed 50/50 with the parametric errors (g12 overfitted with 100 % bank on random tracks).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
KAPPA='{"kappa":0.3,"kappa_corner":0.15}'
GN="--gatenet runs_host/g5/gatenet.pt --ekf3"
SN="--split-noise 3 2 1.5 1.5"
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
BANK=runs_host/resid/bank_a2rl25.pt
# (no wait: g16 dropped by the user on 2026-10-23; refine g15 first, the fast model is trained from the best refined one)
base="--profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/g15 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
# A) polish
[ -f runs/g15p/model.pt ] || $PY -I scripts/train_state.py $base --ent-coef 0 --init-std 0.25 --iters 500 --out runs/g15p > runs/g15p.log 2>&1
# B) g15 camera flights on the replica (seeds 41-44, disjoint from eval seeds) -> track bank
for s in 41 42 43 44; do
  [ -f runs/flight/a25_$s/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g15 --estimator runs_host/e4/estimator.pt \
    --n-envs 24 --steps 3000 --ekf --seed $s $GN --ekf-args "$KAPPA" $SPEC --dump runs/flight/a25_$s > runs/a25_$s.log 2>&1
done
$PY -I scripts/build_resid_bank.py runs/flight/a25_4[1-4]/data.pt --out $BANK > runs/bank_a2rl25.log 2>&1
[ -f runs/g15b/model.pt ] || $PY -I scripts/train_state.py $base --ent-coef 0.001 --init-std 0.35 --iters 1000 --resid-bank $BANK --resid-prob 0.5 --out runs/g15b > runs/g15b.log 2>&1
for r in g15 g15p g15b; do
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body $SPEC $SN --resid-bank "" > runs/x_${r}_spec_sn2.eval 2>&1
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body $SPEC --resid-bank $BANK --resid-prob 1 > runs/x_${r}_spec_bank.eval 2>&1
done
SEEDS="21 22 23 24" scripts/cam_eval.sh g15p_a2rl25x runs/g15p $GN --ekf-args "$KAPPA" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh g15b_a2rl25x runs/g15b $GN --ekf-args "$KAPPA" $SPEC
touch runs/cam_eval/G15X_DONE
