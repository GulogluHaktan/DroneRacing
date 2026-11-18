#!/bin/bash
# Resume after the shutdown of 2026-10-27 ~16:30. Order (user): g36 Isaac target line first, then the rest of g34 seeds 29-36,
# then g16 detector / g33 box. Each step skips work that is already on disk.
# Start: nohup scripts/resume_oct27.sh > runs/resume_oct27.log 2>&1 &
cd /home/haktan/Projects/DroneRacing
# the GPU belongs to this work (user permission): stop a local LLM server that holds it
for p in $(pgrep -x llama-server) $(pgrep -x ollama); do kill "$p" 2>/dev/null; done
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do echo "waiting: GPU memory in use"; sleep 30; done
scripts/run_g36b.sh            # g36 trained already: skips training and the finished surrogate evals, runs Isaac t36_cal seeds 21-28
scripts/run_t34more.sh         # g34 seeds 29-36 not yet in runs/cam_eval/t34_cal.txt (see also t34p_cal.txt from the parallel test)
scripts/run_onpolicy16.sh &    # waits for G36_DONE
scripts/run_g33.sh &           # waits for G36_DONE
wait
touch runs/RESUME_OCT27_DONE
