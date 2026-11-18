#!/bin/bash
# Videos of soup85 (+ g17 detector) on the full target line, seeds 71 and 72.
cd /home/haktan/Projects/DroneRacing
SPEC="--fixed-prob 1 --fixed-jitter 0.3 0.05"
KP='{"kappa":0.3,"kappa_corner":0.15,"aided_perm":true,"n_map":32}'
CROP="--gatenet runs_host/g17/gatenet.pt --ekf3 --crop 384 --derotate --hfov 128 --img-h 616 --img-w 820 --cam-hz 30 --cam-delay 5 --ekf5 --aug-stable --ekf-args $KP $SPEC"
ST="--rs-ms 20 --corrupt 0.1 --exposure-aug 0.4 --band-check --rs-comp 20 --tilt-err 3 --tilt-seed 7 --extr-corr runs_host/calib/cal15_41.pt"
for s in 71; do
  rm -rf runs/video_soup85m_$s; mkdir -p runs/video_soup85m_$s
  flock runs/.isaac.lock scripts/run_isaaclab.sh isaac/eval_vision.py --policy runs/soup85 --estimator runs_host/e4/estimator.pt --n-envs 4 --steps 3000 --ekf --seed $s \
    $CROP $ST --video runs/video_soup85m_$s --video-envs 4 --video-max 4 --drone-model > runs/video_soup85m_$s.log 2>&1
  for d in runs/video_soup85m_$s/*/; do n=$(basename $d); ffmpeg -y -loglevel error -framerate 50 -i $d/%05d.jpg -c:v libx264 -pix_fmt yuv420p -crf 23 runs/video_soup85m_$s/$n.mp4; done
done
touch runs/video_soup85m_DONE
