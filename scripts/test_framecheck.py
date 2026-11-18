"""Corrupt-frame check on own-camera frames (d7) after realistic_augment, with and without isaac/cam_artifacts.corrupt."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "isaac"))
from cam_artifacts import corrupt  # noqa: E402
from droneracing.framecheck import corrupt_frames  # noqa: E402
from droneracing.percep_aug import realistic_augment  # noqa: E402

d = torch.load("runs/percep/d7/data.pt")
idx = torch.randperm(len(d["img"]), generator=torch.Generator().manual_seed(3))[:600]
g = torch.Generator(device="cuda").manual_seed(5)
import itertools
fp = tp = 0.0
for i in range(0, len(idx), 50):
    img = d["img"][idx[i : i + 50]].cuda().float() / 255.0
    clean = realistic_augment(img, 1.0, gen=g)
    bad = corrupt(clean, 1.0, gen=g)  # every frame corrupted
    fp += corrupt_frames(clean).float().sum().item() / len(idx)
    tp += corrupt_frames(bad).float().sum().item() / len(idx)
print(f"false alarms on clean frames {100*fp:.1f} %  | detected corrupt frames {100*tp:.1f} %")
