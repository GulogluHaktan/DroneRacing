#!/bin/bash
# After g38c (training + surrogate eval) is done and the GPU is idle, finish the fresh-seed confirmation of g37 (seeds 80-86; 71-79 are on disk).
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/G38C_DONE ]; do sleep 30; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 20; done
rm -f runs/cam_eval/T37C_DONE
scripts/run_g37_confirm.sh
