#!/bin/bash
# After g38h (training + surrogate eval) is done: Isaac eval of g38h on seeds 71-78 (compare with t38d_cal: 24/32 = 75.0 %, 19.12 m/s, velocity error 0.445).
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/G38H_DONE ]; do sleep 30; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 20; done
rm -f runs/cam_eval/t38h_cal_DONE
POL=runs/g38h TAG=t38h_cal scripts/run_g38_isaac.sh
