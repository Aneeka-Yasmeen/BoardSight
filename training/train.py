import argparse

from ultralytics import YOLO

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--name", default="pcb")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.resume:
        YOLO(a.base).train(resume=True)
        raise SystemExit
    YOLO(a.base).train(
        data=a.data, project=a.project, name=a.name, exist_ok=True, epochs=a.epochs, batch=a.batch, imgsz=640,
        patience=15, workers=0, cos_lr=True, close_mosaic=10, degrees=8, scale=0.4, translate=0.1, flipud=0.5, fliplr=0.5,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.35, mosaic=0.5, seed=7, plots=True, amp=True, device=0,
    )
