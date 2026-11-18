#!/bin/bash
# 155 deg / 308x410 GateNet2 (g9) from d4, then the camera line
cd /home/haktan/Projects/DroneRacing
/home/haktan/isaac-env/bin/python -I scripts/train_gatenet2.py --data runs/percep/d4/data.pt --init runs_host/g5/gatenet.pt --epochs 8 --bs 16 --out runs_host/g9 > runs/fov_train.log 2>&1
scripts/cam_eval.sh fov155 runs/g6 --gatenet runs_host/g9/gatenet.pt --ekf3 --hfov 155 --img-h 308 --img-w 410 --ekf-args '{"sig_px_corner":4.8,"assoc_px_corner":48.0,"dup_px":9.6}'
