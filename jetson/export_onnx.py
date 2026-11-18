#!/usr/bin/env python3
"""Export the gate networks to ONNX for a TensorRT speed test on the Jetson (runs on the training PC).

  ~/isaac-env/bin/python -I jetson/export_onnx.py
Outputs jetson/onnx/gatenet_g5_384.onnx (GateNet on a 384x384 crop) and jetson/onnx/gatenetmr_m0b_384.onnx (MonoRace mask net).
On the Jetson:  /usr/src/tensorrt/bin/trtexec --onnx=gatenet_g5_384.onnx --fp16 --useCudaGraph
Budget at 85 fps: 11.7 ms per frame for crop + network + corner extraction + filter.
"""
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from droneracing.gatenet import GateNet  # noqa: E402
from droneracing.monorace import GateNetMR  # noqa: E402

out = root / "jetson" / "onnx"
out.mkdir(parents=True, exist_ok=True)
x = torch.zeros(1, 3, 384, 384)

ck = torch.load(root / "runs_host/g5/gatenet.pt", map_location="cpu")
g = GateNet(refine=ck.get("refine", True)).eval()
g.load_state_dict(ck["net"])
torch.onnx.export(g, x, out / "gatenet_g5_384.onnx", opset_version=17, input_names=["img"], dynamo=False)

mk = torch.load(root / "runs_host/m0b/monorace.pt", map_location="cpu")
m = GateNetMR(mk.get("f", 4)).eval()
m.load_state_dict(mk["net"])


class _Last(torch.nn.Module):  # deployment uses only the full-resolution output (MonoRace)
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x)[-1])


torch.onnx.export(_Last(m), x, out / "gatenetmr_m0b_384.onnx", opset_version=17, input_names=["img"], output_names=["mask"], dynamo=False)
print("written:", *sorted(p.name for p in out.glob("*.onnx")))
