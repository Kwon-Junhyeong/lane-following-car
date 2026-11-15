#!/usr/bin/env python3
"""Run the detector and the label rules of laptop/detector_server.py on recorded onboard video.

Each frame goes through the same path as on the car: the RGB array from the camera is
JPEG-encoded at quality 85 by car/main.py, decoded on the laptop, passed to YOLO, and
reduced to one label by select_label(). With --true-color the frame is sent in BGR order.

usage: python check/replay_detector.py best.pt VIDEO_OR_DIR [...] > detector.csv
"""
import os
import sys
import glob
import argparse

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "laptop"))
from detector_server import select_label, CONF_TH, IMG_SIZE
from ultralytics import YOLO


def list_videos(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "*.avi"), recursive=True))
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--true-color", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    names = model.model.names if hasattr(model.model, "names") else model.names
    print("clip,frame,selected,satisfied,detections")
    for path in list_videos(args.videos):
        clip = os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path)
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            sent = bgr if args.true_color else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ok, jpg = cv2.imencode(".jpg", sent, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            r = model.predict(frame, conf=CONF_TH, imgsz=IMG_SIZE, verbose=False)[0]
            selected, satisfied = select_label(r.boxes, names)
            dets = []
            if r.boxes is not None:
                for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    dets.append(f"{names[int(c)]}:{p:.2f}:{b[2] - b[0]:.0f}")
            sat = " ".join(d["class"] for d in satisfied)
            print(f"{clip},{i},{selected or ''},{sat},{' '.join(dets)}")
            i += 1
        cap.release()


if __name__ == "__main__":
    main()
