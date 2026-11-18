#!/bin/bash
# Close-range perception (Isaac loop: 1-3 m found only 44-55 %): flight-like close views (0.8-3.5 m, nose-down 20-60 deg, roll up to
# 45 deg, lateral offset), own camera 820x616 with 384 crops -> d10; g15 = g14 fine-tuned on d9 + d10 (realcam, derotate).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while [ ! -f runs/cam_eval/DIAG14B_DONE ]; do sleep 60; done
[ -f runs/percep/d10/data.pt ] || flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/collect_percep.py --task runs/g17 --frames 12000 --hfov 128 --img-h 616 --img-w 820 \
  --crop-out 384 --extr-deg 3 --n-envs 4 --dist 0.8 3.5 --pitch-range 20 60 --roll-max 45 --yaw-noise 0.5 --lateral 1.0 --inverted-frac 0.05 --seed 31 --preview 4 --out runs/percep/d10 > runs/d10_collect.log 2>&1
[ -f runs_host/g15/gatenet.pt ] || flock runs/.isaac.lock $PY -I scripts/train_gatenet2.py --data runs/percep/d9/data.pt runs/percep/d10/data.pt --init runs_host/g14/gatenet.pt --epochs 6 --bs 16 --lr 3e-4 --realcam --derotate 3 --holdout --out runs_host/g15 > runs/g15_train.log 2>&1
{ echo "== g14 on d10 (close)"; $PY -I scripts/percep_breakdown.py runs_host/g14/gatenet.pt --data runs/percep/d10/data.pt --n 1000 --derotate 3
  echo "== g15 on d10 (close)"; $PY -I scripts/percep_breakdown.py runs_host/g15/gatenet.pt --data runs/percep/d10/data.pt --n 1000 --derotate 3
  echo "== g15 on d9"; $PY -I scripts/percep_breakdown.py runs_host/g15/gatenet.pt --data runs/percep/d9/data.pt --n 1000 --derotate 3; } > runs/percep_breakdown_g15.txt 2>&1
touch runs/CLOSE15_DONE
