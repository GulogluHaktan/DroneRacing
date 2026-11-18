#!/bin/bash
# after the crop render-health reruns: old 130 deg line with the new gate-obs error line (seeds 21-23, 24 drones)
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/RH_DONE ]; do sleep 60; done
SEEDS="21 22 23" scripts/cam_eval.sh base_gerr runs/g6 --gatenet runs_host/g5/gatenet.pt --ekf3
touch runs/cam_eval/GERR_DONE
