#!/usr/bin/env python3
"""Closed-loop lane simulation with ceiling-light reflections on the floor.

usage: python sim/lane_sim.py OUT.csv
       python sim/lane_sim.py --frame GAIN GLARE S OUT.png
"""
import os
import sys
import json
import math
import argparse
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "car"))
sys.dont_write_bytecode = True

from modules.steering_module import SteeringAnalyzer  # noqa: E402
from modules import fsm_module  # noqa: E402

# camera
F_PX = 250.0             # focal length, px
CX, CY = 320.0, 240.0
CAM_H = 0.12             # height above the floor, m
PITCH = math.radians(23.8)
CAM_X = 0.15             # ahead of the rear axle, m

# track
LANE_W = 0.226           # line center to line center, m
LINE_W = 0.018           # m
STRAIGHT = 2.0           # m
RADIUS = 0.6             # m
LINES = [-1.5 * LANE_W, -0.5 * LANE_W, 0.5 * LANE_W, 1.5 * LANE_W]

# vehicle
WHEELBASE = 0.16         # m
MAX_STEER = math.radians(25.0)   # front wheel angle at full lock
SPEED = {80: 0.30, 60: 0.22}     # m/s for the motor values the code sends
DT = 1.0 / 30.0                  # one camera frame

SERVO = {"LEFT": 33.0, "RIGHT": 147.0}
PWM = {"FORWARD": 80, "LEFT": 60, "RIGHT": 60}
STRAIGHT_SERVO = 78.5

CAL = json.load(open(os.path.join(HERE, "calibration.json")))


def _rays():
    v, u = np.mgrid[190:390, 0:640].astype(np.float64)
    dx, dy = (u - CX) / F_PX, (v - CY) / F_PX
    fwd = np.cos(PITCH) - dy * np.sin(PITCH)
    down = np.sin(PITCH) + dy * np.cos(PITCH)
    t = CAM_H / down
    return (t * fwd).astype(np.float32), (-t * dx).astype(np.float32), t.astype(np.float32)   # forward, left, range


FWD, LEFT, RANGE = _rays()
NOISE = np.random.default_rng(7).normal(0, 3.0, (8, 200, 640, 3)).astype(np.float32)   # sensor noise, sigma 3


def signed_distance(x, y):
    """Signed distance from the lane center line of the oval (positive outward)."""
    half = STRAIGHT / 2.0
    cx = np.clip(x, -half, half)
    return np.hypot(x - cx, y) - RADIUS


def pose_on_track(s):
    """Point and heading of the lane center at arc length s (counterclockwise, starting mid bottom straight)."""
    half = STRAIGHT / 2.0
    per = 2 * STRAIGHT + 2 * math.pi * RADIUS
    s = s % per
    if s < half:
        return s, -RADIUS, 0.0
    s -= half
    if s < math.pi * RADIUS:
        a = -math.pi / 2 + s / RADIUS
        return half + RADIUS * math.cos(a), RADIUS * math.sin(a), a + math.pi / 2
    s -= math.pi * RADIUS
    if s < STRAIGHT:
        return half - s, RADIUS, math.pi
    s -= STRAIGHT
    if s < math.pi * RADIUS:
        a = math.pi / 2 + s / RADIUS
        return -half + RADIUS * math.cos(a), RADIUS * math.sin(a), a + math.pi / 2
    s -= math.pi * RADIUS
    return -half + s, -RADIUS, 0.0


def make_glare(level, seed):
    """Reflections as floor-fixed elliptic patches spread over the track; level 1 = density fitted to the recordings."""
    rng = np.random.default_rng(seed)
    per = 2 * STRAIGHT + 2 * math.pi * RADIUS
    n = int(round(level * CAL["glare_per_m"] * per))
    blobs = []
    for _ in range(n):
        x, y, th = pose_on_track(rng.uniform(0, per))
        off = rng.uniform(-2.0, 2.0) * LANE_W
        bx, by = x - off * math.sin(th), y + off * math.cos(th)
        a = rng.uniform(*CAL["glare_axis_m"])
        b = a * rng.uniform(0.35, 0.8)
        blobs.append((bx, by, a, b, rng.uniform(0, math.pi), rng.uniform(*CAL["glare_peak_v"])))
    return blobs


def render(x, y, th, gain, blobs, rng):
    """BGR camera frame (640 x 480) for a car whose rear axle is at (x, y) with heading th."""
    cxw, cyw = x + CAM_X * math.cos(th), y + CAM_X * math.sin(th)
    c, s = math.cos(th), math.sin(th)
    wx = cxw + FWD * c - LEFT * s
    wy = cyw + FWD * s + LEFT * c
    d = signed_distance(wx, wy)
    soft = 0.0015 * RANGE                                # about one pixel of floor at that range
    cover = np.zeros_like(d)
    for off in LINES:
        e = np.abs(d - off) - LINE_W / 2
        cover = np.maximum(cover, np.clip(0.5 - e / soft, 0, 1))
    floor = np.array(CAL["floor_bgr"], np.float32)
    line = np.array(CAL["line_bgr"], np.float32)
    cover = cover.astype(np.float32)
    img = floor[None, None, :] * (1 - cover[..., None]) + line[None, None, :] * cover[..., None]
    # reflections on every 4th pixel, then upsampled
    gx, gy = wx[2::4, 2::4], wy[2::4, 2::4]
    lift = np.zeros(gx.shape, np.float32)
    for bx, by, a, b, ang, peak in blobs:
        if (bx - cxw) ** 2 + (by - cyw) ** 2 > 1.0:
            continue
        ca, sa = math.cos(ang), math.sin(ang)
        px, py = gx - bx, gy - by
        r2 = ((px * ca + py * sa) / a) ** 2 + ((-px * sa + py * ca) / b) ** 2
        lift = np.maximum(lift, (peak / 255.0) * np.exp(-2.0 * r2).astype(np.float32))
    if lift.any():
        lift = cv2.resize(lift, (640, 200), interpolation=cv2.INTER_LINEAR)
        tint = np.array([1.0, CAL["glare_g_over_b"], CAL["glare_r_over_b"]], np.float32) * 255.0
        glare = floor + (tint[None, None, :] - floor) * lift[..., None]
        img = np.maximum(img, glare)
    img = img * gain + NOISE[rng.integers(len(NOISE))]
    band = np.clip(img, 0, 255).astype(np.uint8)
    band = cv2.GaussianBlur(band, (3, 3), 0)
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[190:390] = band
    return frame


def analyzer():
    return SteeringAnalyzer(base_angle=85, max_offset=48.5, angle_limits=(30.0, 150.0),
                            y_bottom_ratio=0.70, y_top_ratio=0.40, k_head=0.60)


def lane_angle(an, frame_bgr, method):
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)        # the car hands the code an RGB array
    if method == "final":
        return an.analyze_frame(frame)["angle"]
    bev = cv2.warpPerspective(frame, an.pre.M, (640, 480))
    gray = cv2.GaussianBlur(cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return an.analyze_binary(binary)["angle"]


def run(args):
    method, gain, level, seed = args
    rng = np.random.default_rng(1000 + seed)
    blobs = make_glare(level, seed)
    an = analyzer()
    per = 2 * STRAIGHT + 2 * math.pi * RADIUS
    x, y, th = pose_on_track(0.0)
    x -= (WHEELBASE / 2) * math.cos(th)
    y -= (WHEELBASE / 2) * math.sin(th)
    travelled, errs, steps = 0.0, [], 0
    while travelled < per and steps < 3000:
        frame = render(x, y, th, gain, blobs, rng)
        angle = lane_angle(an, frame, method)
        cmd = fsm_module._drive_by_angle(angle - 5)
        servo = SERVO.get(cmd, angle - 5)
        v = SPEED[PWM[cmd]]
        lim = (147.0 - STRAIGHT_SERVO) if servo >= STRAIGHT_SERVO else (STRAIGHT_SERVO - 33.0)
        delta = -MAX_STEER * (servo - STRAIGHT_SERVO) / lim        # servo above 78.5 steers right
        x += v * math.cos(th) * DT
        y += v * math.sin(th) * DT
        th += v / WHEELBASE * math.tan(delta) * DT
        travelled += v * DT
        steps += 1
        mx, my = x + (WHEELBASE / 2) * math.cos(th), y + (WHEELBASE / 2) * math.sin(th)
        e = float(signed_distance(np.array(mx), np.array(my)))
        errs.append(e)
        if abs(e) > LANE_W / 2:
            break
    errs = np.abs(np.array(errs))
    done = travelled >= per
    return dict(method=method, gain=gain, glare=level, seed=seed, completed=int(done),
                distance_m=round(travelled, 3), lap_fraction=round(min(travelled / per, 1.0), 4),
                mean_abs_err_cm=round(100 * errs.mean(), 2), max_abs_err_cm=round(100 * errs.max(), 2), steps=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frame", nargs=3, type=float, metavar=("GAIN", "GLARE", "S"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.frame:
        gain, level, s = a.frame
        x, y, th = pose_on_track(s)
        cv2.imwrite(a.out, render(x, y, th, gain, make_glare(level, 0), np.random.default_rng(0)))
        return
    gains = [0.80, 0.85, 0.90, 1.00, 1.20, 1.45]
    levels = [0.0, 1.0, 2.0]
    jobs = [(m, g, l, s) for m in ("final", "grayscale") for g in gains for l in levels for s in range(3)]
    with Pool(a.workers) as p:
        rows = p.map(run, jobs)
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
