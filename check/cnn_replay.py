#!/usr/bin/env python3
"""Run the three-class lane CNN (cnn/train.py) on recorded onboard video, one CSV row per frame.

The Keras weights are read from the .h5 file with h5py and the network is evaluated with
PyTorch, so TensorFlow is not needed. Preprocessing is the one in cnn/drive.py.

usage: python check/cnn_replay.py MODEL.h5 VIDEO_OR_DIR [...] > cnn.csv
"""
import os
import sys
import glob
import argparse

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cnn"))
import config_preprocessing as cfg_pre

CLASSES = ["FORWARD", "LEFT", "RIGHT"]
M = cv2.getPerspectiveTransform(np.float32(cfg_pre.SRC_POINTS), np.float32(cfg_pre.DST_POINTS))


def load_weights(path):
    w = {}
    with h5py.File(path, "r") as f:
        g = f["model_weights"]
        for layer in ["conv2d", "conv2d_1", "conv2d_2", "dense", "dense_1"]:
            sub = g[layer]["sequential"][layer]
            w[layer] = (torch.from_numpy(sub["kernel"][()]), torch.from_numpy(sub["bias"][()]))
    return w


def forward(w, x):
    for layer in ["conv2d", "conv2d_1", "conv2d_2"]:
        k, b = w[layer]
        x = F.relu(F.conv2d(x, k.permute(3, 2, 0, 1), b))
        x = F.max_pool2d(x, 2)
    x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
    k, b = w["dense"]
    x = F.relu(x @ k + b)
    k, b = w["dense_1"]
    return torch.softmax(x @ k + b, dim=1)


def preprocess(frame):
    bev = cv2.warpPerspective(frame, M, cfg_pre.BEV_SIZE)
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    if cfg_pre.USE_BLUR:
        gray = cv2.GaussianBlur(gray, cfg_pre.BLUR_KERNEL, 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def to_input(frame):
    binary = preprocess(frame)
    resized = cv2.resize(binary, (128, 128))
    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    x = torch.from_numpy((resized / 255.0).astype(np.float32))
    return x.permute(2, 0, 1)


def list_videos(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.avi")))
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("videos", nargs="+")
    args = ap.parse_args()

    w = load_weights(args.model)
    print("clip,frame,cnn_class,p_forward,p_left,p_right")
    with torch.no_grad():
        for path in list_videos(args.videos):
            clip = os.path.splitext(os.path.basename(path))[0]
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(to_input(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)))
            cap.release()
            for s in range(0, len(frames), 64):
                p = forward(w, torch.stack(frames[s:s + 64])).numpy()
                for j, row in enumerate(p):
                    print(f"{clip},{s + j},{CLASSES[int(row.argmax())]},{row[0]:.4f},{row[1]:.4f},{row[2]:.4f}")


if __name__ == "__main__":
    main()
