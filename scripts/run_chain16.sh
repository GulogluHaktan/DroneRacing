#!/bin/bash
# KAIST-style perception: YOLOv8-pose gate keypoints trained on the same d6 frames as GateSeg, then the camera line on the 2025 replica
# (g15p + aided_perm), same seeds as g15p_perm (GateNet) and g15p_seg (GateSeg). Waits for the MonoRace chain, GPU must be idle.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/cam_eval/SEG_DONE ]; do sleep 60; done
[ -f runs_host/y0/yolopose.pt ] || $PY -I scripts/train_yolopose.py --data runs/percep/d6/data.pt --imgsz 256 --epochs 40 --out runs_host/y0 > runs/y0_train.log 2>&1
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true}'
SEEDS="21 22 23 24" scripts/cam_eval.sh g15p_yolo runs/g15p --gatenet runs_host/g5/gatenet.pt --ekf3 --yolo runs_host/y0/yolopose.pt --yolo-imgsz 256 --ekf-args "$KP" $SPEC
touch runs/cam_eval/YOLO_DONE
