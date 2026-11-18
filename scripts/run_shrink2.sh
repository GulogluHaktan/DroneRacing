#!/bin/bash
# Offset-box trial (MonoRace trains with a gate opening much smaller than the real one): g17 fine-tuned with gate shrink 0.20 / 0.40 on the own-drone profile.
# Evaluation always uses the physical opening (gate body). Control: g17 itself (runs/own_g17_*.eval), same eval flags.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SN="--split-noise 3 2 1.5 1.5"; SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
base="--profile own_drone --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --v-max-start 13 --v-max 13 --finish-time 75 $SN $SPEC"
for s in 20 40; do
  n=g2${s:0:1}s
  [ -f runs/$n/model.pt ] || $PY -I scripts/train_state.py $base --init runs/g17 --ent-coef 0.001 --init-std 0.3 --iters 1200 --gate-shrink 0.$s --out runs/$n > runs/$n.log 2>&1
  $PY -I scripts/eval_state.py --run runs/$n --n-envs 4096 --gate-body $SPEC $SN --resid-bank "" --profile own_drone > runs/own_${n}.eval 2>&1
done
touch runs/SHRINK2_DONE
