#!/usr/bin/env python3
"""IMU from the flight controller over MSP (Betaflight), stamped with the arrival time on the Jetson (CLOCK_MONOTONIC).

KAIST (arXiv 2512.20475) uses the arrival time because the flight-controller clock drifts against the Jetson clock;
Kalibr then gives the constant camera-IMU offset. This script polls MSP_RAW_IMU (102) as fast as the link allows,
prints the rate and the jitter, and logs (t, acc xyz, gyro xyz) to a .npy file for Kalibr / the filter.

Usage on the Jetson: python3 msp_imu.py --port /dev/ttyTHS1 --baud 1000000 --seconds 10 --out imu.npy
Betaflight: enable MSP on the UART; raise the baud rate; set the gyro / accel rates. NOT TESTED ON HARDWARE.
Units: Betaflight MSP_RAW_IMU gives raw accel in 1/512 g (acc_1G = 512 on most targets; check yours) and gyro in deg/s.
"""
import argparse
import struct
import time

import numpy as np
import serial

MSP_RAW_IMU = 102

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="/dev/ttyTHS1")
ap.add_argument("--baud", type=int, default=1000000)
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--acc-1g", type=float, default=512.0)
ap.add_argument("--out", default="imu.npy")
a = ap.parse_args()


def request(cmd: int) -> bytes:
    return b"$M<" + bytes([0, cmd, cmd])  # size 0, checksum = size ^ cmd


def read_reply(s: serial.Serial):
    if s.read(3) != b"$M>":
        s.reset_input_buffer()
        return None, None
    size, cmd = s.read(2)
    payload = s.read(size)
    chk = s.read(1)
    c = size ^ cmd
    for b in payload:
        c ^= b
    if len(chk) != 1 or chk[0] != c:
        return None, None
    return cmd, payload


s = serial.Serial(a.port, a.baud, timeout=0.01)
rows = []
t_end = time.monotonic() + a.seconds
while time.monotonic() < t_end:
    s.write(request(MSP_RAW_IMU))
    cmd, pl = read_reply(s)
    t = time.clock_gettime(time.CLOCK_MONOTONIC)
    if cmd == MSP_RAW_IMU and pl is not None and len(pl) >= 12:
        ax, ay, az, gx, gy, gz = struct.unpack("<6h", pl[:12])
        rows.append((t, ax / a.acc_1g * 9.81, ay / a.acc_1g * 9.81, az / a.acc_1g * 9.81, np.radians(gx), np.radians(gy), np.radians(gz)))
d = np.array(rows)
np.save(a.out, d)
if len(d) > 2:
    dt = np.diff(d[:, 0]) * 1000
    print(f"samples {len(d)}  rate {1000 / np.median(dt):.1f} Hz  period p50/p99/max {np.median(dt):.2f}/{np.percentile(dt, 99):.2f}/{dt.max():.2f} ms")
    print(f"accel norm at rest (expect 9.81): {np.linalg.norm(d[:, 1:4], axis=1).mean():.2f} m/s^2")
else:
    print("no MSP_RAW_IMU replies: check the port, the baud rate and that MSP is enabled on this UART")
