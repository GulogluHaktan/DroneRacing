#!/bin/bash
# Resume after the shutdown of 2026-10-26 (see handover.md): finish the GateNetMR training if needed, then test the MonoRace line in the camera loop.
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 20; done
if [ ! -f runs_host/m0/DONE ]; then
  # continue from the last saved epoch with a smaller learning rate (the schedule is not restored)
  cp runs_host/m0/monorace.pt runs_host/m0/monorace_part.pt
  $PY -I scripts/train_monorace_net.py --data runs/percep/d8/data.pt --epochs 15 --lr 2e-4 --init runs_host/m0/monorace_part.pt --out runs_host/m0b > runs/m0b_train.log 2>&1
  cp runs_host/m0b/monorace.pt runs_host/m0/monorace.pt; touch runs_host/m0/DONE
fi
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
B="--gatenet runs_host/g5/gatenet.pt --mr runs_host/m0/monorace.pt --mr-crop 192 --hfov 128 --img-h 308 --img-w 410 --cam-hz 30 --cam-delay 5 --aug-stable"
# first a smoke test (1 seed, 12 drones): look at runs/mr_smoke.log for errors before the long runs
NENV=12 SEEDS="21" scripts/cam_eval.sh mr_smoke runs/g17 $B --ekf-args "$KP" $SPEC
NENV=12 SEEDS="21 22 23 24 41 42 43 44" scripts/cam_eval.sh mr_stable30 runs/g17 $B --ekf-args "$KP" $SPEC
NENV=12 SEEDS="21 22 23 24" scripts/cam_eval.sh mr_stress_all runs/g17 $B --rs-ms 20 --corrupt 0.1 --tilt-err 3 --ekf-args "$KP" $SPEC
touch runs/cam_eval/MR_DONE
