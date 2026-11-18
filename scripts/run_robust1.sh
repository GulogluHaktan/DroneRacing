#!/bin/bash
# Train against the realism stresses (rolling shutter 20 ms, 10 % corrupt frames, 3 deg unknown mount error) instead of the clean simulator.
#  A. GateNet g12 = g5 fine-tuned on own-camera data d7 with rolling-shutter augmentation (0-30 ms, labels shifted with their row).
#  B. Camera line under the combined stress, seeds 21-24: g17 + g5 (earlier: 16.7 %) versus g17 + g12, + corrupt-frame check (--frame-check); clean line as regression check.
#  C. Stressed closed-loop dumps (seeds 41-44) -> residual bank -> g26 = g17 + 50 % stress bank -> torch check -> camera line under stress and clean.
# Waits for the torch trainers of the box / shrink trials. Marker runs/ROBUST1_DONE.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/BOX1_DONE ] || [ ! -f runs/SHRINK2_DONE ] || [ ! -f runs/cam_eval/CAMMODE_DONE ]; do sleep 60; done  # camera-mode test first
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
ST="--rs-ms 20 --corrupt 0.1 --tilt-err 3"
cam() { echo "--gatenet $1 --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"; }
# A
[ -f runs_host/g12/gatenet.pt ] || $PY -I scripts/train_gatenet2.py --data runs/percep/d7/data.pt --init runs_host/g5/gatenet.pt --epochs 8 --bs 16 --rs-ms-max 30 --out runs_host/g12 > runs/g12_train.log 2>&1
# B
SEEDS="21 22 23 24" scripts/cam_eval.sh rob_stress_g12 runs/g17 $(cam runs_host/g12/gatenet.pt) $ST
SEEDS="21 22 23 24" scripts/cam_eval.sh rob_stress_g12_fc runs/g17 $(cam runs_host/g12/gatenet.pt) $ST --frame-check
SEEDS="21 22 23 24" scripts/cam_eval.sh rob_clean_g12 runs/g17 $(cam runs_host/g12/gatenet.pt)
# C
for s in 41 42 43 44; do
  [ -f runs/flight/rob_$s/data.pt ] || { while ps -eo cmd | grep -v grep | grep -q "train_state.py\|train_gatenet"; do sleep 60; done; flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g17 --estimator runs_host/e4/estimator.pt \
    --n-envs 24 --steps 3000 --ekf --seed $s $(cam runs_host/g12/gatenet.pt) $ST --dump runs/flight/rob_$s > runs/rob_dump_$s.log 2>&1; }
done
BANK=runs_host/resid/bank_stress.pt
[ -f $BANK ] || $PY -I scripts/build_resid_bank.py runs/flight/rob_4[1-4]/data.pt --out $BANK > runs/bank_stress.log 2>&1
SN="--split-noise 3 2 1.5 1.5"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g26/model.pt ] || $PY -I scripts/train_state.py $base --init runs/g17 --ent-coef 0.001 --init-std 0.35 --iters 1000 --resid-bank $BANK --resid-prob 0.5 --out runs/g26 > runs/g26.log 2>&1
for r in g17 g26; do
  $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body $SPEC --resid-bank $BANK --resid-prob 1 --profile own_drone > runs/rob_${r}_bank.eval 2>&1
done
SEEDS="21 22 23 24" scripts/cam_eval.sh rob_stress_g26 runs/g26 $(cam runs_host/g12/gatenet.pt) $ST
SEEDS="21 22 23 24" scripts/cam_eval.sh rob_clean_g26 runs/g26 $(cam runs_host/g12/gatenet.pt)
touch runs/ROBUST1_DONE
