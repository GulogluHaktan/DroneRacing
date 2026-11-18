#!/bin/bash
# Which IMX219 mode? Same line (g17, GateNet g5, EKF5, n_map 32, stable appearance, 50 ms delay compensated), seeds 21-24, 12 drones, exact frame rates:
#  A full FOV 1640x1232 @ 30 Hz (stock Jetson mode)        -> 410x308, hfov 128
#  B crop 1280x720 @ 60 Hz (stock mode, assumed 2x2 binning of the 2560x1440 centre: 78 % width, 58 % height, same focal) -> 320x180, hfov 116
#  C full FOV 820x616 @ 90 Hz (modified driver, MonoRace / KAIST) -> 410x308, hfov 128
#  D = C under the combined stress, A under the same stress is runs/cam_eval/stress_rs_corrupt_tilt.txt (16.7 %, 24 drones)
# The crop geometry of mode B is an assumption until the module is on the bench.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
B0="--gatenet runs_host/g5/gatenet.pt --ekf3 --cam-delay 5 --ekf5 --aug-stable --cam-hz-exact --ekf-args $KP $SPEC"
export NENV=12 SEEDS="21 22 23 24"
scripts/cam_eval.sh mode_A30_full runs/g17 $B0 --hfov 128 --img-h 308 --img-w 410 --cam-hz 30
scripts/cam_eval.sh mode_B60_crop runs/g17 $B0 --hfov 116 --img-h 180 --img-w 320 --cam-hz 60
scripts/cam_eval.sh mode_C90_full runs/g17 $B0 --hfov 128 --img-h 308 --img-w 410 --cam-hz 90
scripts/cam_eval.sh mode_C90_stress runs/g17 $B0 --hfov 128 --img-h 308 --img-w 410 --cam-hz 90 --rs-ms 11 --corrupt 0.1 --tilt-err 3
touch runs/cam_eval/CAMMODE_DONE
