#!/bin/bash
# wider camera FOV (MonoRace 155 deg vs our 130): collect frames, fine-tune GateNet2 from g5, camera line with g6 policy; waits for the g8 camera queue
cd /home/haktan/Projects/DroneRacing
while ! grep -q "^DONE" runs/cam_g8.txt 2>/dev/null; do sleep 60; done
if [ ! -f runs/percep/d3/data.pt ]; then
  scripts/run_isaaclab.sh isaac/collect_percep.py --task runs/h100 --frames 24000 --hfov 155 --n-envs 64 --dist 1.2 16 --pitch-range 0 40 --roll-max 35 --yaw-noise 0.7 --lateral 2.5 --inverted-frac 0.03 --seed 7 --out runs/percep/d3 > runs/fov_collect.log 2>&1
fi
/home/haktan/isaac-env/bin/python -I scripts/train_gatenet2.py --data runs/percep/d3/data.pt --init runs_host/g5/gatenet.pt --epochs 8 --out runs_host/g9 > runs/fov_train.log 2>&1
for seed in 21 22 23 24; do
  out=$(scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g6 --gatenet runs_host/g9/gatenet.pt --ekf3 --hfov 155 --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf --seed $seed 2>&1 | grep -E "VISION STACK|estimator in|Traceback|Error|out of memory" | cut -c1-240 | tr '\n' ' ')
  echo "fov155 $seed $out" >> runs/cam_fov.txt
done
echo DONE >> runs/cam_fov.txt
