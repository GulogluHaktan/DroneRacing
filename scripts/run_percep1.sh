#!/bin/bash
# Perception line under real-camera stress (no mount error here; the mount error is handled by calibration, tested separately).
# Stress: rolling shutter 20 ms (rotation), 10 % corrupt frames, 40 % short-exposure frames (dark + gain noise) with lens-edge blur.
# Compare GateNet g5 (clean training) with g13 (realcam training), then g13 + corrupt-frame check, then g13 on the clean line.
cd /home/haktan/Projects/DroneRacing
while ps -eo cmd | grep -v grep | grep -q "train_gatenet2.py"; do sleep 60; done
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
cam() { echo "--gatenet $1 --ekf3 --hfov 128 --img-h 308 --img-w 410 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"; }
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4"
export NENV=12 SEEDS="21 22 23 24"
scripts/cam_eval.sh p1_g5_stress runs/g17 $(cam runs_host/g5/gatenet.pt) $ST
scripts/cam_eval.sh p1_g13_stress runs/g17 $(cam runs_host/g13/gatenet.pt) $ST
scripts/cam_eval.sh p1_g13_stress_fc runs/g17 $(cam runs_host/g13/gatenet.pt) $ST --frame-check
scripts/cam_eval.sh p1_g13_clean runs/g17 $(cam runs_host/g13/gatenet.pt)
scripts/cam_eval.sh p1_g5_clean runs/g17 $(cam runs_host/g5/gatenet.pt)
touch runs/cam_eval/PERCEP1_DONE
