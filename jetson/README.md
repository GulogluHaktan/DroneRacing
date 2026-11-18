# Jetson Orin NX: camera mode, time stamps, IMU, network speed

Tools for the real perception chain (architecture: vault `01-Projects/DroneRacing/droneracing-real-camera-perception-architecture.md`).
Nothing here was tested on hardware. Run the steps in order. Each step tells what to send back.

| File | Purpose | Needs |
|---|---|---|
| `export_onnx.py` | GateNet g5 and GateNetMR m0b as ONNX (384x384 crop) | training PC (done: `onnx/`) |
| `imx219_mode_820x616_85fps.h` | register table: 820x616, full FOV (x4 binning), 85.3 fps | kernel module source |
| `dt_mode_820x616.dtsi` | device-tree mode node for the same mode | camera overlay source |
| `capture_ts.py` | raw V4L2 capture with the start-of-frame time stamp: rate, jitter, dropped frames, FOV frames | camera |
| `led_latency.py` | LED test: rolling-shutter line time and frame-to-user-space delay | camera, LED, resistor |
| `msp_imu.py` | MSP_RAW_IMU from the flight controller with Jetson arrival time: rate, jitter | flight controller (Betaflight, MSP on a UART) |

## Step 0 (now, no camera): back up the Jetson
Make a full image of the NVMe / SD before any kernel or device-tree change. A wrong device tree can stop the boot.
Send back: JetPack / L4T version (`cat /etc/nv_tegra_release`), carrier board name.

## Step 1 (now, no camera): network speed with TensorRT
```
/usr/src/tensorrt/bin/trtexec --onnx=gatenet_g5_384.onnx --fp16 --useCudaGraph
/usr/src/tensorrt/bin/trtexec --onnx=gatenetmr_m0b_384.onnx --fp16 --useCudaGraph
```
Send back: the "GPU Compute Time" mean and p99 of both. Budget at 85 fps: 11.7 ms per frame for everything.

## Step 2 (now, if the flight controller is there): IMU over MSP
Betaflight: enable MSP on the UART to the Jetson, set the highest stable baud rate.
```
python3 msp_imu.py --port /dev/ttyTHS1 --baud 1000000 --seconds 10
```
Send back: rate (target 500 Hz, KAIST), period p99 and max.

## Step 3 (camera arrives): stock modes and FOV
```
v4l2-ctl -d /dev/video0 --list-formats-ext
python3 capture_ts.py --width 1640 --height 1232 --count 300
python3 capture_ts.py --width 1280 --height 720 --count 300 --save fov_720p
python3 capture_ts.py --width 1640 --height 1232 --count 30 --save fov_full
```
Send back: the mode list, rate and jitter per mode, one frame of each FOV folder (gate or a door in view).

## Step 4: add the 820x616 85 fps mode
1. Get the kernel sources of your L4T version (`public_sources.tbz2`, `nvidia-oot`).
2. Add `imx219_mode_820x616_85fps[]` to `imx219_mode_tbls.h`, add it to the mode enum and to the mode table list after the last mode.
3. Add the node of `dt_mode_820x616.dtsi` to the camera overlay of your carrier as the next `modeN` (same index as the table).
4. Build the module and the overlay. Install. Reboot.
5. `python3 capture_ts.py --width 820 --height 616 --count 600` -> expect about 85 fps.
If the image is black or the rate is wrong, send back `dmesg | grep -i imx219` and the v4l2-ctl output.

## Step 5: LED test (camera mode chosen)
```
sudo python3 led_latency.py --pin 7 --width 820 --height 616 --trials 200
```
Move the LED to the top, the middle and the bottom of the image. Send back: line time, readout, delay median and p90.
These numbers go into the simulator (`--rs-ms`, `--cam-delay`) and into the filter (row time, image delay).

## Step 6: Kalibr
Camera intrinsics, camera-IMU rotation (target below 0.5 deg) and the constant time offset, with the IMU log of `msp_imu.py`.
