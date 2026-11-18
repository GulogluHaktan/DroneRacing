#!/bin/bash
cd /home/haktan/Projects/DroneRacing
until grep -q DONE runs/eval_h100.log; do sleep 20; done
exec scripts/run_hz200.sh
