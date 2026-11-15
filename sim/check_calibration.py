#!/usr/bin/env python3
"""Compare synthetic frames (gain 1, recorded-like reflections) with the recordings, through the same pipeline.

Recorded values (check/results/lane_track.csv and the dropped regions of the recordings, every 4th frame):
regions of at least 150 px per frame, share of frames with more than two such regions, area of the regions
that the two-largest rule drops (quartiles, in the warped image), and the HSV of the line and floor pixels.

usage: python sim/check_calibration.py
"""
import os
import sys
import math

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lane_sim as L  # noqa: E402

RECORDED = {"regions_mean": 5.28, "more_than_two": 0.850, "dropped_area_q": (327, 662, 1218),
            "line_v": 231, "line_s": 12, "floor_v": 59, "floor_s": 61}


def main():
    an = L.analyzer()
    pre = an.pre
    per = 2 * L.STRAIGHT + 2 * math.pi * L.RADIUS
    rng = np.random.default_rng(0)
    regs, areas, lv, ls, fv, fs = [], [], [], [], [], []
    for seed in range(4):
        blobs = L.make_glare(1.0, seed)
        for s in np.linspace(0, per, 60, endpoint=False):
            x, y, th = L.pose_on_track(s)
            off = rng.uniform(-0.03, 0.03)
            x, y = x - off * math.sin(th), y + off * math.cos(th)
            th += rng.uniform(-0.15, 0.15)
            bgr = L.render(x - L.WHEELBASE / 2 * math.cos(th), y - L.WHEELBASE / 2 * math.sin(th), th, 1.0, blobs, rng)
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            binary = pre._base_binary(frame)
            kept = pre.process(frame)
            n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            keptlab = set(np.unique(lab[kept > 0])) - {0}
            big = [l for l in range(1, n) if st[l, cv2.CC_STAT_AREA] >= 150]
            regs.append(len(big))
            areas += [int(st[l, cv2.CC_STAT_AREA]) for l in big if l not in keptlab]
            hsv = cv2.cvtColor(cv2.warpPerspective(bgr, pre.M, (640, 480)), cv2.COLOR_BGR2HSV)
            if (kept > 0).any():
                lv.append(np.median(hsv[..., 2][kept > 0])); ls.append(np.median(hsv[..., 1][kept > 0]))
            fv.append(np.median(hsv[..., 2][binary == 0])); fs.append(np.median(hsv[..., 1][binary == 0]))
    regs = np.array(regs)
    print("synthetic: regions_mean %.2f, more_than_two %.3f, dropped_area_q %s, line_v %.0f, line_s %.0f, floor_v %.0f, floor_s %.0f"
          % (regs.mean(), (regs > 2).mean(), [int(np.percentile(areas, p)) for p in (25, 50, 75)] if areas else None,
             np.median(lv), np.median(ls), np.median(fv), np.median(fs)))
    print("recorded:  ", RECORDED)


if __name__ == "__main__":
    main()
