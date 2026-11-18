"""KAIST-style gate detector: YOLOv8-pose with 4 gate keypoints (arXiv 2512.20475), trained on the sim frames of collect_percep.

One object class (gate), box = extent of the four inner corners clipped to the image, keypoints = inner corners TL TR BR BL.
A corner outside the image is unlabeled (visibility 0): YOLO-pose has no extrapolation, unlike GateSeg. Outputs use the GateNet
detection format ([score, 4 corners px] rows) so det_features / PnP / EKF stay unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def export_dataset(data_paths: list[str], out: Path, val_frac: float = 0.05, min_box_px: float = 6.0, seed: int = 0) -> Path:
    """collect_percep data.pt files -> Ultralytics pose dataset (images/labels, train/val) + data.yaml. Returns the yaml path."""
    import cv2

    out = Path(out)
    parts = [torch.load(p, mmap=True) for p in data_paths]
    H, W = parts[0]["hw"]
    rng = np.random.default_rng(seed)
    n_img = 0
    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for d in parts:
        for i in range(len(d["img"])):
            sp = "val" if rng.random() < val_frac else "train"
            cor = d["corners"][i].float().numpy() * np.array([W, H])  # [S,4,2] px
            lines = []
            for s in range(cor.shape[0]):
                if not bool(d["slot_valid"][i, s]):
                    continue
                c = cor[s]
                inside = (c[:, 0] >= 0) & (c[:, 0] < W) & (c[:, 1] >= 0) & (c[:, 1] < H)
                if not inside.any():
                    continue
                x0, y0 = np.clip(c.min(0), 0, [W, H]); x1, y1 = np.clip(c.max(0), 0, [W, H])
                if max(x1 - x0, y1 - y0) < min_box_px:
                    continue
                kp = " ".join(f"{c[k, 0] / W:.5f} {c[k, 1] / H:.5f} 2" if inside[k] else "0 0 0" for k in range(4))
                lines.append(f"0 {(x0 + x1) / 2 / W:.5f} {(y0 + y1) / 2 / H:.5f} {(x1 - x0) / W:.5f} {(y1 - y0) / H:.5f} {kp}")
            name = f"{n_img:07d}"
            cv2.imwrite(str(out / "images" / sp / f"{name}.jpg"), d["img"][i].permute(1, 2, 0).numpy()[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / sp / f"{name}.txt").write_text("\n".join(lines))
            n_img += 1
    yml = out / "data.yaml"
    yml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nkpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\nnames:\n  0: gate\n")
    return yml


@torch.no_grad()
def detect(model, img: torch.Tensor, imgsz: int, conf: float = 0.25):
    """Image batch [B,3,H,W] float 0..1 -> list of [n,9] tensors (score + 4 corners px, TL TR BR BL)."""
    res = model.predict(img, imgsz=imgsz, conf=conf, verbose=False, half=False)
    out = []
    for r in res:
        if r.keypoints is None or len(r.boxes) == 0:
            out.append(torch.zeros(0, 9))
            continue
        kp = r.keypoints.xy.cpu().reshape(-1, 8)
        out.append(torch.cat((r.boxes.conf.cpu()[:, None], kp), dim=1).float())
    return out
