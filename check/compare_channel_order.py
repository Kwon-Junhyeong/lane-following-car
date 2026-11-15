#!/usr/bin/env python3
"""Run the lane pipeline on every recorded frame twice: in the channel order the car uses, and with red and blue swapped.

The car hands the camera's RGB array to code written for BGR (cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2GRAY).
check/replay_lane.py reproduces that order. This script also runs the pipeline on the same frame in BGR order
and compares the binary image before region selection, the kept image, and the lane angle.

usage: python check/compare_channel_order.py RECORDINGS_DIR > channel_order.csv
       (a one-line summary is printed to stderr)
"""
import os
import sys
import glob

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "car"))

from modules.steering_module import SteeringAnalyzer


def make_analyzer():
    return SteeringAnalyzer(base_angle=85, max_offset=48.5, angle_limits=(30.0, 150.0),
                            y_bottom_ratio=0.70, y_top_ratio=0.40, k_head=0.60)


def main():
    an = make_analyzer()
    pre = an.pre
    print("clip,frame,binary_diff_px,kept_diff_px,angle_car,angle_swapped")
    n = n_bin = n_kept = n_angle = 0
    max_angle = 0.0
    for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.avi"))):
        clip = os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path)
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            car = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)   # camera order, as in replay_lane.py
            swp = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)   # red and blue swapped against the car
            b1, b2 = pre._base_binary(car), pre._base_binary(swp)
            k1, k2 = pre.process(car), pre.process(swp)
            a1, a2 = an.analyze_binary(k1)["angle"], an.analyze_binary(k2)["angle"]
            db = int(np.count_nonzero(b1 != b2))
            dk = int(np.count_nonzero(k1 != k2))
            print(f"{clip},{i},{db},{dk},{a1:.2f},{a2:.2f}")
            n += 1
            n_bin += db > 0
            n_kept += dk > 0
            if abs(a1 - a2) > 1e-9:
                n_angle += 1
                max_angle = max(max_angle, abs(a1 - a2))
            i += 1
        cap.release()
    print(f"frames {n}; binary differs on {n_bin}; kept image differs on {n_kept}; "
          f"lane angle differs on {n_angle}, by at most {max_angle:.2f} degrees", file=sys.stderr)


if __name__ == "__main__":
    main()
