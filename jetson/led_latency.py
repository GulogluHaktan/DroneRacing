#!/usr/bin/env python3
"""LED test: camera delay and rolling-shutter line time.

An LED on a Jetson GPIO pin switches on at known CLOCK_MONOTONIC times. For every switch-on, the first frame that shows
the LED gives (frame start time stamp, image row of the LED). Over many trials:
  t_on - t_SOF = row * line_time - exposure_offset   (linear fit over the LED rows of the trials)
  -> slope = line time (readout of one row), the full readout = line_time * height.
Move the LED to several heights in the image (top, middle, bottom) to cover the rows.
The delay from the frame start to the time the frame is in user space is printed too (arrival - t_SOF).

Wiring: LED + 330 ohm resistor from the GPIO pin to GND. Point the camera at the LED, dark room, fixed exposure.
Usage on the Jetson: sudo python3 led_latency.py --pin 7 --width 820 --height 616 --trials 200
Needs Jetson.GPIO and OpenCV with V4L2. NOT TESTED ON HARDWARE.
"""
import argparse
import time

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dev", type=int, default=0)
ap.add_argument("--pin", type=int, default=7, help="BOARD pin number of the LED")
ap.add_argument("--width", type=int, default=820)
ap.add_argument("--height", type=int, default=616)
ap.add_argument("--trials", type=int, default=200)
ap.add_argument("--thr", type=float, default=60.0, help="brightness step that marks the LED")
a = ap.parse_args()

import Jetson.GPIO as GPIO  # noqa: E402

GPIO.setmode(GPIO.BOARD)
GPIO.setup(a.pin, GPIO.OUT, initial=GPIO.LOW)
cap = cv2.VideoCapture(a.dev, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

rows, dts, arrive = [], [], []
ref = None
for k in range(a.trials):
    GPIO.output(a.pin, GPIO.LOW)
    for _ in range(4):  # LED off frames, reference image
        ok, fr = cap.read()
    ref = fr.astype(np.float32).reshape(a.height, -1)
    time.sleep(np.random.uniform(0.0, 0.012))  # random phase against the frame clock
    t_on = time.clock_gettime(time.CLOCK_MONOTONIC)
    GPIO.output(a.pin, GPIO.HIGH)
    for _ in range(4):
        ok, fr = cap.read()
        t_arr = time.clock_gettime(time.CLOCK_MONOTONIC)
        t_sof = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # V4L2 buffer time stamp (start of frame)
        diff = fr.astype(np.float32).reshape(a.height, -1) - ref
        rmax = diff.max(axis=1)
        hit = np.nonzero(rmax > a.thr)[0]
        if len(hit):
            rows.append(int(hit[0])); dts.append(t_on - t_sof); arrive.append(t_arr - t_sof)
            break
GPIO.output(a.pin, GPIO.LOW)
GPIO.cleanup()

rows, dts = np.array(rows), np.array(dts)
print(f"LED seen in {len(rows)} of {a.trials} trials")
if len(rows) > 10 and rows.max() - rows.min() > 50:
    A = np.stack((rows, np.ones_like(rows)), 1)
    (slope, icpt), *_ = np.linalg.lstsq(A, dts, rcond=None)
    print(f"line time {slope * 1e6:.2f} us -> readout of {a.height} rows {slope * a.height * 1000:.2f} ms; offset {icpt * 1000:.2f} ms")
else:
    print("not enough row spread: move the LED to other image heights and run again")
print(f"frame start -> user space: median {np.median(arrive) * 1000:.2f} ms, p90 {np.percentile(arrive, 90) * 1000:.2f} ms")
