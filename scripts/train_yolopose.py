"""Train the KAIST-style YOLOv8-pose gate detector on collect_percep data (droneracing/yolopose.py).
Run: ~/isaac-env/bin/python -I scripts/train_yolopose.py --data runs/percep/d6/data.pt --imgsz 256 --epochs 40 --out runs_host/y0
Without --weights the network trains from scratch (yolov8n-pose.yaml, no download).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from droneracing.yolopose import export_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=256, help="256 for 256x192 frames, 384 for the 384 crops")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--model", default="yolov8n-pose.yaml", help="yaml (scratch) or .pt (pretrained)")
    ap.add_argument("--out", default="runs_host/y0")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    ds = out / "dataset"
    yml = ds / "data.yaml" if (ds / "data.yaml").exists() else export_dataset(a.data, ds)
    from ultralytics import YOLO

    m = YOLO(a.model)
    # fixed camera, no geometric flips of the sim frames except left-right (flip_idx swaps the corners); HSV kept (exposure varies)
    m.train(data=str(yml), imgsz=a.imgsz, epochs=a.epochs, batch=a.bs, project=str(out), name="train", exist_ok=True, amp=False,
            flipud=0.0, fliplr=0.5, mosaic=0.5, degrees=0.0, workers=4, plots=False, pretrained=a.model.endswith(".pt"))
    best = out / "train" / "weights" / "best.pt"
    (out / "yolopose.pt").write_bytes(best.read_bytes())
    print("saved", out / "yolopose.pt")


if __name__ == "__main__":
    main()
