#!/bin/bash
# Time box (user leaves at ~16:30 on 2026-11-03): stop the g37 confirmation after seed 79 is done, or at 15:40 at the latest,
# then release the video script by touching T37C_DONE. The pooled success counts the seeds that finished.
cd /home/haktan/Projects/DroneRacing
LIMIT=$(date -d "2026-11-03 15:40" +%s)
while ! grep -q "^79 " runs/cam_eval/t37c_cal.txt 2>/dev/null && [ "$(date +%s)" -lt "$LIMIT" ]; do sleep 20; done
# stop the confirmation chain (pattern is safe here: this script's own command line does not contain it)
pkill -f "run_g37_confirm.sh"
pkill -f "cam_eval.sh t37c_cal"
for c in $(docker ps --format '{{.Names}}' | grep '^dr_'); do docker stop "$c" >/dev/null 2>&1; done
sleep 5
# an unfinished seed leaves no VISION line (cam_eval writes after the run); drop any error/blank line of a killed seed
sed -i '/^[0-9]* 20[0-9][0-9]-/d' runs/cam_eval/t37c_cal.txt
touch runs/cam_eval/T37C_DONE
echo "confirmation stopped at $(date): $(grep -c VISION runs/cam_eval/t37c_cal.txt) valid seeds"
