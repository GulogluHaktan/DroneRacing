#!/bin/bash
# Resume after the shutdown of 2026-11-03 ~12:30. Order: g37 Isaac eval first (skips seeds already in runs/cam_eval/t37_cal.txt; the
# policy is trained already), then the remaining failing-seed dumps (skips runs/flight/f17_*/data.pt that exist).
# Start: nohup scripts/resume_nov3.sh > runs/resume_nov3.log 2>&1 &
cd /home/haktan/Projects/DroneRacing
# the GPU belongs to this work (user permission): stop a local LLM server that holds it
for p in $(pgrep -x llama-server) $(pgrep -x ollama); do kill "$p" 2>/dev/null; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do echo "waiting: GPU memory in use"; sleep 30; done
rm -f runs/cam_eval/T37_DONE runs/FAILSEEDS_DONE
scripts/run_g37.sh            # trained already (model.pt + "it 1000" in runs/g37.log): runs only the missing eval seeds
scripts/run_failseeds.sh      # dumps of seeds 36 35 43 44 22 23 25, skips the ones on disk
touch runs/RESUME_NOV3_DONE
