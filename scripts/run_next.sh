#!/bin/bash
# after the GT-ablation queue: (a) EKF4 corner-count noise, (b) 2x render/network resolution, (c) 155 deg FOV (collect + fine-tune + eval). Strictly sequential: no trainer may overlap a camera eval.
cd /home/haktan/Projects/DroneRacing
while ! grep -q "^DONE" runs/cam_gt.txt 2>/dev/null; do sleep 30; done
EV="scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g6 --estimator runs_host/e4/estimator.pt --n-envs 24 --steps 3000 --ekf"
run() { # tag seed args...
  local tag=$1 seed=$2; shift 2
  out=$($EV "$@" --seed $seed 2>&1 | grep -E "VISION STACK|estimator in|Traceback|Error|out of memory" | cut -c1-240 | tr '\n' ' ')
  echo "$tag $seed $out" >> runs/cam_next.txt
}
# more seeds for the GT ablation (is velocity alone worth a VIO?)
for seed in 23 24; do
  run base $seed --gatenet runs_host/g5/gatenet.pt --ekf3
  run gt_vel $seed --gatenet runs_host/g5/gatenet.pt --ekf3 --ablate vel --ablate-gt
  run gt_vel+up $seed --gatenet runs_host/g5/gatenet.pt --ekf3 --ablate vel,up --ablate-gt
  run gt_all $seed --gatenet runs_host/g5/gatenet.pt --ekf3 --ablate vel,up,gate --ablate-gt
done
echo "ABL2 DONE" >> runs/cam_next.txt
for seed in 21 22 23; do run ekf4 $seed --gatenet runs_host/g5/gatenet.pt --ekf4; done
for seed in 21 22; do run res2x $seed --gatenet runs_host/g5/gatenet.pt --ekf3 --img-h 384 --img-w 512 --ekf-args '{"sig_px_corner":6.0,"assoc_px_corner":60.0,"dup_px":12.0}'; done
echo "PHASE1 DONE" >> runs/cam_next.txt
if [ ! -f runs/percep/d3/data.pt ]; then
  scripts/run_isaaclab.sh isaac/collect_percep.py --task runs/h100 --frames 24000 --hfov 155 --n-envs 64 --dist 1.2 16 --pitch-range 0 40 --roll-max 35 --yaw-noise 0.7 --lateral 2.5 --inverted-frac 0.03 --seed 7 --out runs/percep/d3 > runs/fov_collect.log 2>&1
fi
/home/haktan/isaac-env/bin/python -I scripts/train_gatenet2.py --data runs/percep/d3/data.pt --init runs_host/g5/gatenet.pt --epochs 8 --out runs_host/g9 > runs/fov_train.log 2>&1
for seed in 21 22 23; do run fov155 $seed --gatenet runs_host/g9/gatenet.pt --ekf3 --hfov 155; done
echo DONE >> runs/cam_next.txt
