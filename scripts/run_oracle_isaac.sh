#!/bin/bash
# (1) g17 flown in Isaac with the env's analytic observation (oracle: no camera, no filter): the part of the gap that is NOT perception.
# (2) closed-loop dumps of g17 on the own camera line (30 Hz, 50 ms, EKF5) on seeds 41-42 for scripts/crash_anatomy.py.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
# perfect sensing in Isaac physics: the policy sees ground-truth velocity, attitude and gates (everything else as the camera line); the ceiling of policy + Isaac dynamics
SEEDS="21 22 23 24" scripts/cam_eval.sh gt_ablate_g17 runs/g17 --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ablate vel,up,gate --ablate-gt --ekf-args "$KP" $SPEC
for s in 41 42 43 44; do
  [ -f runs/flight/own_$s/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g17 --estimator runs_host/e4/estimator.pt \
    --n-envs 24 --steps 3000 --ekf --seed $s --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args "$KP" $SPEC --dump runs/flight/own_$s > runs/own_dump_$s.log 2>&1
done
# (3) Swift-style bank from the own-camera line (seeds 41-44 are disjoint from the eval seeds 21-24), g19 = g17 + 50 % bank, torch checks, camera line
PY=/home/haktan/isaac-env/bin/python
BANK=runs_host/resid/bank_own30.pt
[ -f $BANK ] || $PY -I scripts/build_resid_bank.py runs/flight/own_4[1-4]/data.pt --out $BANK > runs/bank_own30.log 2>&1
SN="--split-noise 3 2 1.5 1.5"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g19/model.pt ] || $PY -I scripts/train_state.py $base --init runs/g17 --ent-coef 0.001 --init-std 0.35 --iters 1000 --resid-bank $BANK --resid-prob 0.5 --out runs/g19 > runs/g19.log 2>&1
for r in g17 g19; do
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body $SPEC --resid-bank $BANK --resid-prob 1 --profile own_drone > runs/own_${r}_bank.eval 2>&1
done
SEEDS="21 22 23 24" scripts/cam_eval.sh own_g19_comp30 runs/g19 --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args "$KP" $SPEC
touch runs/cam_eval/ORACLE_DONE
