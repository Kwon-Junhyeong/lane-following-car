#!/usr/bin/env python3
"""Run the car's lane pipeline on recorded onboard video, one CSV row per frame.

Analyzer settings are those of car/main.py; the command is fsm_module._drive_by_angle(angle - 5),
as in the lane-following states. regions and angle_unfiltered: white regions of at least 150 px,
and the angle computed from all of them, before the two largest are kept.

usage: python check/replay_lane.py VIDEO_OR_DIR [...] > lane.csv
"""
import os
import sys
import time
import glob
import argparse

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "car"))

from modules.steering_module import SteeringAnalyzer
from modules import fsm_module

SERVO = {"LEFT": 33.0, "RIGHT": 147.0}


def make_analyzer():
    return SteeringAnalyzer(
        base_angle=85,
        max_offset=48.5,
        angle_limits=(30.0, 150.0),
        y_bottom_ratio=0.70,
        y_top_ratio=0.40,
        k_head=0.60,
    )


def unfiltered(analyzer, frame):
    binary = analyzer.pre._base_binary(frame)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions = int((stats[1:, cv2.CC_STAT_AREA] >= analyzer.pre.min_area).sum())
    return regions, analyzer.analyze_binary(binary)["angle"]


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
    ap.add_argument("videos", nargs="+")
    args = ap.parse_args()

    analyzer = make_analyzer()
    print("clip,frame,c_bot,c_top,angle,command,servo,ms,regions,angle_unfiltered")
    for path in list_videos(args.videos):
        clip = os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path)
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            t0 = time.perf_counter()
            r = analyzer.analyze_frame(frame)
            ms = (time.perf_counter() - t0) * 1000.0
            angle = r["angle"]
            pts = r["points"]
            cmd = fsm_module._drive_by_angle(angle - 5)
            servo = SERVO.get(cmd, angle - 5)
            cb = "" if pts["c_bot"] is None else pts["c_bot"]
            ct = "" if pts["c_top"] is None else pts["c_top"]
            regions, raw_angle = unfiltered(analyzer, frame)
            print(f"{clip},{i},{cb},{ct},{angle:.2f},{cmd},{servo:.2f},{ms:.2f},{regions},{raw_angle:.2f}")
            i += 1
        cap.release()


if __name__ == "__main__":
    main()
