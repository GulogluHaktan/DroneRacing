#!/bin/bash
# EKF gate-offset smoothing: kappa 0.6 (default) follows the newest PnP almost fully; test stronger averaging on the old line (seeds 21-23, 24 drones)
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/GERR_DONE ]; do sleep 60; done
SEEDS="21 22 23" scripts/cam_eval.sh kappa30 runs/g6 --gatenet runs_host/g5/gatenet.pt --ekf3 --ekf-args '{"kappa":0.3,"kappa_corner":0.15}'
SEEDS="21 22 23" scripts/cam_eval.sh kappa15 runs/g6 --gatenet runs_host/g5/gatenet.pt --ekf3 --ekf-args '{"kappa":0.15,"kappa_corner":0.08}'
touch runs/cam_eval/KAPPA_DONE
