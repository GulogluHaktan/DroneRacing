#!/bin/bash
# Videos of g37 + g17 on the full target line (same flags as the evaluation). Starts only after the fresh-seed confirmation (t37c_cal) is
# done AND the pooled success (t37_cal seeds 21-44 + t37c_cal seeds 71-86, 128 flights) is >= 85 %. Both finished and crashed clips are kept.
cd /home/haktan/Projects/DroneRacing
while [ ! -f runs/cam_eval/T37C_DONE ]; do sleep 60; done
read pooled fresh <<<$(~/isaac-env/bin/python -I - <<'PY'
import re
def tally(f):
    s=0;n=0
    for l in open(f):
        m=re.search(r"success\s+([\d.]+)%",l)
        if m: s+=round(float(m.group(1))/25);n+=4
    return s,n
a=tally("runs/cam_eval/t37_cal.txt"); b=tally("runs/cam_eval/t37c_cal.txt")
print(f"{100*(a[0]+b[0])/(a[1]+b[1]):.1f} {100*b[0]/b[1]:.1f}")
PY
)
echo "pooled 128-flight success: $pooled % | fresh seeds only: $fresh %"
if ! awk -v p="$pooled" 'BEGIN{exit !(p>=85.0)}'; then echo "below 85 %: no video"; touch runs/cam_eval/VIDEOG37_SKIPPED; exit 0; fi
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g17/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do sleep 20; done
for s in 71 72; do
  rm -rf runs/video_g37_$s; mkdir -p runs/video_g37_$s
  flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/g37 --estimator runs_host/e4/estimator.pt --n-envs 4 --steps 3000 --ekf --seed $s \
    $CROP $ST --video runs/video_g37_$s --video-envs 4 --video-max 4 > runs/video_g37_$s.log 2>&1
  for d in runs/video_g37_$s/*/; do n=$(basename $d); ffmpeg -y -loglevel error -framerate 50 -i $d/%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 23 runs/video_g37_$s/$n.mp4; done
done
touch runs/cam_eval/VIDEOG37_DONE
