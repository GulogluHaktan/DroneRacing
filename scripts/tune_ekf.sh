#!/bin/bash
cd /home/haktan/Projects/DroneRacing
for cfg in '{}' '{"sig_acc":0.3}' '{"sig_acc":1.2}' '{"sig_pnp":[0.15,0.06]}' '{"sig_pnp":[0.4,0.15]}' '{"sig_acc":0.3,"sig_pnp":[0.15,0.06]}' '{"chi2":25.0}' '{"kappa":0.9}'; do
  echo "### $cfg"
  ~/isaac-env/bin/python -I scripts/test_ekf.py --run runs/t8snap --n-envs 256 --steps 800 --ekf-args "$cfg" 2>&1 | grep -E "^ ?[0-9]+-|next-gate rel|EKF-OL"
done
echo DONE
