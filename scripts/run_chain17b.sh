#!/bin/bash
# Chain 17 restarted with a fixed per-drone appearance (--aug-stable) in every camera evaluation: the per-frame colour flicker of the test augmentation is not in a real fixed-exposure camera.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'  # n_map 32: the default 16 gate slots make the gates of lap 2 share one correction (replay: lap-2 gate error p50 0.4-0.5 -> 0.06-0.10 m)
T="--cam-hz 30 --cam-delay 5 --ekf5 --aug-stable"
# (0) n_map 32 only (flicker kept): compare with 63.5 % (n_map 16, flicker)
SEEDS="21 22 23 24" scripts/cam_eval.sh own_g17_nmap32 runs/g17 --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --ekf-args "$KP" $SPEC
# (1) n_map 32 + stable appearance (GateNet g5, 256x192)
SEEDS="21 22 23 24" scripts/cam_eval.sh own_g17_stable30 runs/g17 --gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 $T --ekf-args "$KP" $SPEC
# (1c) which realism item breaks the line: one at a time, seeds 21-24, base filter
B1="--gatenet runs_host/g5/gatenet.pt --ekf3 --hfov 128 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable"
SEEDS="21 22 23 24" scripts/cam_eval.sh stress_tilt3 runs/g17 $B1 --tilt-err 3 --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh stress_rs20 runs/g17 $B1 --rs-ms 20 --ekf-args "$KP" $SPEC
SEEDS="21 22 23 24" scripts/cam_eval.sh stress_corrupt10 runs/g17 $B1 --corrupt 0.1 --ekf-args "$KP" $SPEC
# (2) MonoRace mask, own camera frames 410x308 (trained after the n_map checks: the GPU is needed for the camera evaluations first)
[ -f runs_host/s1/gateseg.pt ] || $PY -I scripts/train_gateseg.py --data runs/percep/d7/data.pt --epochs 8 --out runs_host/s1 > runs/s1_train.log 2>&1
L="--ekf3 --hfov 128 --img-h 308 --img-w 410 $T"
NENV=12 SEEDS="21 22 23 24 41 42 43 44" scripts/cam_eval.sh d7_gateseg_stable runs/g17 --gatenet runs_host/g5/gatenet.pt --gateseg runs_host/s1/gateseg.pt $L --ekf-args "$KP" $SPEC
[ -f runs_host/g11/gatenet.pt ] || $PY -I scripts/train_gatenet2.py --data runs/percep/d7/data.pt --init runs_host/g5/gatenet.pt --epochs 10 --bs 16 --out runs_host/g11 > runs/g11_train.log 2>&1
NENV=12 SEEDS="21 22 23 24 41 42 43 44" scripts/cam_eval.sh d7_gatenet2_stable runs/g17 --gatenet runs_host/g11/gatenet.pt $L --ekf-args "$KP" $SPEC
[ -f runs_host/y1/yolopose.pt ] || $PY -I scripts/train_yolopose.py --data runs/percep/d7/data.pt --imgsz 416 --epochs 40 --out runs_host/y1 > runs/y1_train.log 2>&1
NENV=12 SEEDS="21 22 23 24 41 42 43 44" scripts/cam_eval.sh d7_yolo_stable runs/g17 --gatenet runs_host/g11/gatenet.pt --yolo runs_host/y1/yolopose.pt --yolo-imgsz 416 $L --ekf-args "$KP" $SPEC
touch runs/cam_eval/CHAIN17_DONE
