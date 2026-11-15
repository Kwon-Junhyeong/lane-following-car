#!/usr/bin/env python3
"""Summarize sim/results_lane_sim.csv and write the data of docs/sim.png.

For each preprocessing, reflection level and brightness gain: laps completed (of the seeds), mean share of the
lap driven before leaving the lane, mean lateral error. The gain is also given as the mean HSV value of the
camera band at that gain (reflections at the recorded level), for comparison with the recordings (V 54 to 101).

usage: python sim/summarize_sim.py sim/results_lane_sim.csv OUT_DIR
"""
import os
import sys
import math

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lane_sim as L  # noqa: E402


def band_v(gain):
    rng = np.random.default_rng(0)
    blobs = L.make_glare(1.0, 0)
    per = 2 * L.STRAIGHT + 2 * math.pi * L.RADIUS
    vs = []
    for s in np.linspace(0, per, 24, endpoint=False):
        x, y, th = L.pose_on_track(s)
        f = L.render(x, y, th, gain, blobs, rng)
        vs.append(cv2.cvtColor(f[190:390], cv2.COLOR_BGR2HSV)[..., 2].mean())
    return float(np.mean(vs))


def main():
    d = pd.read_csv(sys.argv[1])
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    g = d.groupby(["method", "glare", "gain"]).agg(
        runs=("seed", "size"), completed=("completed", "sum"),
        lap_pct=("lap_fraction", lambda s: 100 * s.mean()), err_cm=("mean_abs_err_cm", "mean")).reset_index()
    vmap = {gain: band_v(gain) for gain in sorted(d["gain"].unique())}
    g["band_v"] = g["gain"].map(vmap).round(1)
    print(g.to_string(index=False))
    for (m, lvl), x in g.groupby(["method", "glare"]):
        x[["band_v", "lap_pct", "err_cm", "completed", "runs"]].to_csv(
            os.path.join(out, f"sim_{m}_{int(lvl)}.dat"), sep=" ", index=False)
    g.to_csv(os.path.join(HERE, "results_lane_sim_summary.csv"), index=False)


if __name__ == "__main__":
    main()
