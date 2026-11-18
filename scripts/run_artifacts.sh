#!/bin/bash
# Phase 1a: cost of each A2RL camera artefact on the camera line (g6, GateNet2+EKF3), 4 seeds each; waits for the perception queue
cd /home/haktan/Projects/DroneRacing
true
B="--gatenet runs_host/g5/gatenet.pt --ekf3"
scripts/cam_eval.sh art_base runs/g6 $B
scripts/cam_eval.sh art_delay2 runs/g6 $B --cam-delay 2
scripts/cam_eval.sh art_rs10 runs/g6 $B --rs-ms 10
scripts/cam_eval.sh art_corrupt20 runs/g6 $B --corrupt 0.2
scripts/cam_eval.sh art_tilt2 runs/g6 $B --tilt-err 2
scripts/cam_eval.sh art_all runs/g6 $B --cam-delay 2 --rs-ms 10 --corrupt 0.2 --tilt-err 2
for f in runs/cam_eval/art_*.txt; do echo "$f"; done > runs/cam_eval/ART_DONE
