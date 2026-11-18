#!/bin/bash
# Isaac eval with the new TRUE mean-speed metric ("mean-speed" in the VISION line, mean speed of the finished flights) on seeds 71-78:
# g38c first (path-speed 18.5 in the surrogate: likely the best chance for success > 75 % with mean speed >= 20), then g38d again (for the mean-speed).
cd /home/haktan/Projects/DroneRacing
rm -f runs/cam_eval/t38c_cal_DONE runs/cam_eval/t38d2_cal_DONE
POL=runs/g38c TAG=t38c_cal scripts/run_g38_isaac.sh
POL=runs/g38d TAG=t38d2_cal scripts/run_g38_isaac.sh
touch runs/SPEEDCAND_DONE
