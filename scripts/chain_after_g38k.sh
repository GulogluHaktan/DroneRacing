#!/bin/bash
# After g38k (training + surrogate eval): Isaac eval on seeds 71-78 (g38d on the same seeds: 24/32 twice, true mean speed 21.1 m/s).
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/G38K_DONE ]; do sleep 30; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 20; done
rm -f runs/cam_eval/t38k_cal_DONE
POL=runs/g38k TAG=t38k_cal scripts/run_g38_isaac.sh
