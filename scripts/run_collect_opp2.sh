#!/bin/bash
# wide-pass data: 100 % of the frames use the whole frame letterboxed into the 384 window; opponents 1.5-10 m
cd /home/haktan/Projects/DroneRacing
flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/collect_opp.py --task runs/o3 --frames 8000 --hfov 128 --img-h 616 --img-w 820 --crop-out 384 --extr-deg 3 --n-envs 8 --dist 1.0 18 --pitch-range 0 45 --roll-max 60 --yaw-noise 0.7 --lateral 2.5 --inverted-frac 0.10 --opp-dist 1.5 10 --wide-frac 1.0 --seed 47 --out runs/percep/o2 > runs/o2_collect.log 2>&1
touch runs/O2_COLLECT_DONE
