#!/bin/bash
cd /home/haktan/Projects/DroneRacing
for s in 31 32 33; do
  echo "== collect seed $s"
  scripts/run_isaaclab.sh isaac/collect_flight.py --task runs/t8snap --gatenet runs_host/g2/gatenet.pt --n-envs 24 --steps 3500 --seed $s --out runs/flight/f$s 2>&1 | grep -E "FLIGHT|Traceback|Error" | head -3
  ls -la runs/flight/f$s 2>&1 | tail -2
done
echo DONE
