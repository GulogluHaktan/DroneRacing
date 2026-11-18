#!/bin/bash
# Own-camera detector comparison (MonoRace / KAIST / our GateNet2) on the SAME frames: 128 deg, 410x308 (half of 820x616), 45 deg tilt.
# d7 -> GateSeg s1 (MonoRace mask + edge-line corners), GateNet2 g11 (init g5), YOLOv8-pose y1 (KAIST) -> camera line 30 Hz, 50 ms delay, EKF5, g17.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/cam_eval/VIDEO_DONE ]; do sleep 60; done
[ -f runs/percep/d7/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/collect_percep.py --task runs/g17 --frames 24000 --hfov 128 --img-h 308 --img-w 410 \
  --n-envs 24 --dist 1.0 18 --pitch-range 0 45 --roll-max 60 --yaw-noise 0.7 --lateral 2.5 --inverted-frac 0.10 --extr-deg 3 --seed 17 --preview 4 --out runs/percep/d7 > runs/d7_collect.log 2>&1
[ -f runs_host/s1/gateseg.pt ] || $PY -I scripts/train_gateseg.py --data runs/percep/d7/data.pt --epochs 12 --out runs_host/s1 > runs/s1_train.log 2>&1
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
L="--ekf3 --hfov 128 --img-h 308 --img-w 410 --cam-hz 30 --cam-delay 5 --ekf5"
# MonoRace style first (user priority 2026-10-25)
SEEDS="21 22 23 24" scripts/cam_eval.sh d7_gateseg runs/g17 --gatenet runs_host/g5/gatenet.pt --gateseg runs_host/s1/gateseg.pt $L --ekf-args "$KP" $SPEC
[ -f runs_host/g11/gatenet.pt ] || $PY -I scripts/train_gatenet2.py --data runs/percep/d7/data.pt --init runs_host/g5/gatenet.pt --epochs 10 --bs 16 --out runs_host/g11 > runs/g11_train.log 2>&1
[ -f runs_host/y1/yolopose.pt ] || $PY -I scripts/train_yolopose.py --data runs/percep/d7/data.pt --imgsz 416 --epochs 40 --out runs_host/y1 > runs/y1_train.log 2>&1
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
L="--ekf3 --hfov 128 --img-h 308 --img-w 410 --cam-hz 30 --cam-delay 5 --ekf5"
SEEDS="21 22 23 24" scripts/cam_eval.sh d7_gatenet2 runs/g17 --gatenet runs_host/g11/gatenet.pt $L --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh d7_yolo runs/g17 --gatenet runs_host/g11/gatenet.pt --yolo runs_host/y1/yolopose.pt --yolo-imgsz 416 $L --ekf-args "$KP" $SPEC
touch runs/cam_eval/CHAIN17_DONE
