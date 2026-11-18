#!/usr/bin/env python3
"""Raw V4L2 capture on the Jetson with the driver time stamp of every frame (no ISP, no Argus queue).

The Tegra VI driver stamps each buffer at the start of frame (CLOCK_MONOTONIC). This is the capture time that the
filter needs (KAIST: hardware acquisition time stamp, not the arrival time in the processing queue).
Prints the frame rate, the jitter of the frame period and the dropped frames; optionally saves frames (debayered) to
check the field of view of a mode.

Usage on the Jetson (camera on /dev/video0):
  v4l2-ctl -d /dev/video0 --list-formats-ext                      # list the driver modes
  python3 capture_ts.py --width 820 --height 616 --count 600       # measure rate + jitter
  python3 capture_ts.py --width 1280 --height 720 --count 60 --save fov_720p   # frames to check the FOV
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dev", default="/dev/video0")
ap.add_argument("--width", type=int, required=True)
ap.add_argument("--height", type=int, required=True)
ap.add_argument("--fmt", default="RG10", help="V4L2 pixel format of the sensor mode (RAW10 bayer)")
ap.add_argument("--count", type=int, default=600)
ap.add_argument("--exposure-us", type=int, default=1500, help="fixed exposure (0 = leave the driver value)")
ap.add_argument("--gain", type=int, default=0, help="fixed gain in driver units (0 = leave)")
ap.add_argument("--save", default="", help="directory for debayered PNG frames (first 10 frames)")
a = ap.parse_args()

ctrl = []
if a.exposure_us:
    ctrl.append(f"exposure={a.exposure_us}")
if a.gain:
    ctrl.append(f"gain={a.gain}")
if ctrl:
    subprocess.run(["v4l2-ctl", "-d", a.dev, "-c", ",".join(ctrl)], check=False)

raw = Path("/tmp/capture_ts.raw")
cmd = ["v4l2-ctl", "-d", a.dev, f"--set-fmt-video=width={a.width},height={a.height},pixelformat={a.fmt}",
       "--stream-mmap", f"--stream-count={a.count}", "--verbose", f"--stream-to={raw}"]
p = subprocess.run(cmd, capture_output=True, text=True)
log = p.stdout + p.stderr
ts = np.array([float(m.group(1)) for m in re.finditer(r"ts:\s*([\d.]+)", log)])
seq = np.array([int(m.group(1)) for m in re.finditer(r"seq:\s*(\d+)", log)])
if len(ts) < 3:
    print("no time stamps found; v4l2-ctl output:\n" + log[-2000:])
    sys.exit(1)
dt = np.diff(ts) * 1000.0
print(f"frames {len(ts)}  rate {1000.0 / np.median(dt):.2f} fps  period median {np.median(dt):.3f} ms  "
      f"p1/p99 {np.percentile(dt, 1):.3f}/{np.percentile(dt, 99):.3f} ms  max {dt.max():.3f} ms")
if len(seq) == len(ts):
    print(f"dropped frames (sequence gaps): {int((np.diff(seq) - 1).sum())}")

if a.save:
    import cv2
    out = Path(a.save)
    out.mkdir(parents=True, exist_ok=True)
    data = np.fromfile(raw, dtype=np.uint16)
    stride = data.size // len(ts)
    for i in range(min(10, len(ts))):
        fr = data[i * stride: i * stride + a.width * a.height].reshape(a.height, a.width)
        img8 = (np.clip(fr, 0, 1023) >> 2).astype(np.uint8)
        bgr = cv2.cvtColor(img8, cv2.COLOR_BayerRG2BGR)
        cv2.imwrite(str(out / f"frame_{i:02d}.png"), bgr)
    print(f"saved {min(10, len(ts))} frames to {out}")
