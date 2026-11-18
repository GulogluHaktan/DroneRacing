#!/bin/bash
# Resume after the shutdown of 2026-11-04 ~13:35: finish the Isaac eval of g38d + g19 (seeds 72-78; 71 is on disk; d14 and g19 are done).
# Start: nohup scripts/resume_nov4.sh > runs/resume_nov4.log 2>&1 &
cd /home/haktan/Projects/DroneRacing
for p in $(pgrep -x llama-server) $(pgrep -x ollama); do kill "$p" 2>/dev/null; done   # the GPU belongs to this work (user permission)
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do echo "waiting: GPU memory in use"; sleep 30; done
rm -f runs/cam_eval/T38D19_DONE
scripts/run_g19.sh     # skips the data collection (d14 on disk) and the training (runs_host/g19/gatenet.pt exists); cam_eval skips seeds already in t38d19_cal.txt
touch runs/RESUME_NOV4_DONE
