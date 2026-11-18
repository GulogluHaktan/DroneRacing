#!/usr/bin/env python3
"""Export the current detector (default g17) to ONNX, 384x384 crop, CPU only (no GPU needed).
  ~/isaac-env/bin/python -I jetson/export_g17.py [runs_host/g17/gatenet.pt]
Output: jetson/onnx/gatenet_<name>_384.onnx. Then on the Jetson: trtexec --onnx=... --fp16 --useCudaGraph"""
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from droneracing.gatenet import GateNet  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "runs_host/g17/gatenet.pt"
name = src.parent.name
ck = torch.load(src, map_location="cpu")
g = GateNet(refine=ck.get("refine", True)).eval()
g.load_state_dict(ck["net"])
out = root / "jetson" / "onnx"
out.mkdir(parents=True, exist_ok=True)
x = torch.zeros(1, 3, 384, 384)
torch.onnx.export(g, x, out / f"gatenet_{name}_384.onnx", opset_version=17, input_names=["img"], dynamo=False)
with torch.no_grad():
    y = g(x)
print("exported", out / f"gatenet_{name}_384.onnx", "output shapes:", [tuple(t.shape) for t in (y if isinstance(y, (list, tuple)) else [y])])
