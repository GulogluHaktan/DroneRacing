#!/bin/bash
# Reward study (raise the policy ceiling): g6 finishes only ~71 % even with perfect perception and has 28 % crashes in training
# at 17 m/s; crash penalty 10 is ~4 % of the episode return (~255) while finish_time 150 pays for speed.
# r0 = control (same reward, more training), r1-r3 = safer reward variants. All fine-tuned from g6 for 1000 its, then torch eval
# with the body-size gate on 4096 drones (crash causes printed). Run only with the GPU free of Isaac jobs.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while docker ps -q --filter ancestor=lunarsim-isaaclab:6.0.1 | grep -q .; do echo "waiting: Isaac on GPU"; sleep 60; done
base="--profile a2rl_hall --hall-prob 0.8 --drag-prob 0.25 --n-envs 4096 --log-every 100 --hide-params --map-obs --no-obstacles --sensor-profile ekf --sensor-scale 0.6 --obs-history 8 --n-obs-gates 3 --init-ng 3 --init runs/g6 --split-prob 0.15 --bias-rand 0.3 2 --gamma 0.995 --policy-hz 100 --gate-body --ent-coef 0.001 --soft-frac 0 --v-max-start 30 --v-max 30 --iters 1000"
$PY -I scripts/eval_state.py --run runs/g6 --n-envs 4096 --gate-body > runs/reward_g6.eval 2>&1
for spec in "r0|--finish-time 150" "r1|--finish-time 150 --crash-pen 40" "r2|--finish-time 75 --crash-pen 30" "r3|--finish-time 150 --crash-pen 30 --center-rew 3"; do
  tag=${spec%%|*}; extra=${spec#*|}
  [ -f runs/$tag/model.pt ] || $PY -I scripts/train_state.py $base $extra --out runs/$tag > runs/$tag.log 2>&1
  $PY -I scripts/eval_state.py --run runs/$tag --n-envs 4096 --gate-body > runs/reward_$tag.eval 2>&1
done
touch runs/REWARD_DONE
