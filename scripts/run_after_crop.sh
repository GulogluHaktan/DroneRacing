#!/bin/bash
# paired baseline for crop384 (same 12 seeds x 8 drones, old camera 130 deg 256x192, g5 GateNet2) and then the artefact costs
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/CROP_DONE ]; do sleep 60; done
NENV=8 SEEDS="21 22 23 24 25 26 27 28 29 30 31 32" scripts/cam_eval.sh base8 runs/g6 --gatenet runs_host/g5/gatenet.pt --ekf3
rm -f runs/cam_eval/art_base.txt
scripts/run_artifacts.sh
