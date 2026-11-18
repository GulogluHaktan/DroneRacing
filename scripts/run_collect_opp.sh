#!/bin/bash
# opponent-labelled crop data (g17 crop recipe + scripted opponents inside the crop window)
cd /home/haktan/Projects/DroneRacing
flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/collect_opp.py --task runs/o3 --frames 12000 --hfov 128 --img-h 616 --img-w 820 --crop-out 384 --extr-deg 3 --n-envs 8 --dist 1.0 18 --pitch-range 0 45 --roll-max 60 --yaw-noise 0.7 --lateral 2.5 --inverted-frac 0.10 --opp-dist 1.5 16 --seed 31 --out runs/percep/o1 > runs/o1_collect.log 2>&1
touch runs/O1_COLLECT_DONE
