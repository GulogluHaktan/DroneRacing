#!/bin/bash
# Flight loss is mostly at gate 4 of the 2025 replica (entry angle 24 deg p50, 15 of 27 first-episode hits, belief error ~0.2 m).
# g26 = g17 fine-tuned with a gate-entry angle penalty (--align-pen 6, free up to 10 deg). Torch check vs g17, then Isaac with g15 perception.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
[ -f runs/g26/model.pt ] && grep -q "it 1000" runs/g26.log 2>/dev/null || flock runs/.isaac.lock $PY -I scripts/train_state.py $base --init runs/g17 --ent-coef 0.001 --init-std 0.3 --iters 1000 --align-pen 6 --out runs/g26 > runs/g26.log 2>&1
for r in g17 g26; do
  flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body $SPEC $SN --resid-bank "" --profile own_drone > runs/align_${r}_param.eval 2>&1
  flock runs/.isaac.lock $PY -I scripts/eval_state.py --run runs/$r --n-envs 4096 --gate-body $SPEC --resid-bank runs_host/resid/bank_own30.pt --resid-prob 1 --profile own_drone > runs/align_${r}_bank.eval 2>&1
done
touch runs/ALIGN_TORCH_DONE
SPEC2="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g15/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC2"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20"
NENV=4 SEEDS="21 22 23 24 25 26 27 28" scripts/cam_eval.sh g26_g15_stress runs/g26 $CROP $ST
touch runs/cam_eval/ALIGN_DONE
