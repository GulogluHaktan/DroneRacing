#!/bin/bash
# Policy sensitivity to the perception error of our own camera line: g17 under the closed-loop residual bank bank_own30 scaled by s
# (s = 1: today's error, s = 0: perfect perception). Then the Isaac perception tests (run_percep1.sh).
cd /home/haktan/Projects/DroneRacing
PY=/home/haktan/isaac-env/bin/python
while ps -eo cmd | grep -v grep | grep -q "train_gatenet2.py\|calib_extrinsic.py"; do sleep 30; done
for s in 1.0 0.75 0.5 0.35 0.25 0.0; do
  echo "scale $s"; $PY -I scripts/eval_state.py --run runs/g17 --n-envs 4096 --gate-body --fixed-prob 1 --fixed-jitter 0.3 0.05 --resid-bank runs_host/resid/bank_own30.pt --resid-prob 1 --resid-scale $s --profile own_drone 2>&1 | grep -E " ALL |rror"
done > runs/sweep_bank_scale.txt
exec scripts/run_percep1.sh
