#!/usr/bin/env python3
"""Summarize the CSV written by replay_lane.py, and optionally the one written by cnn_replay.py.

usage: python check/summarize_lane.py lane.csv [cnn.csv]
"""
import sys

import pandas as pd

STRAIGHT_SERVO = 80.0


def group_of(clip):
    return clip.rsplit("_", 1)[0]


def main():
    d = pd.read_csv(sys.argv[1])
    d["group"] = d["clip"].map(group_of)
    d["found"] = d["c_bot"].notna() & d["c_top"].notna()

    print("frames", len(d), "clips", d["clip"].nunique())
    print("both scan rows hit a lane pixel: %.1f %%" % (100 * d["found"].mean()))
    print("pipeline time per frame on this PC: median %.1f ms, 95th percentile %.1f ms"
          % (d["ms"].median(), d["ms"].quantile(0.95)))

    lr = d[d["group"].isin(["left", "right"])].copy()
    lr["side"] = "straight"
    lr.loc[lr["servo"] < STRAIGHT_SERVO, "side"] = "left"
    lr.loc[lr["servo"] > STRAIGHT_SERVO, "side"] = "right"
    print("\nsteering side in the clips filed as left and right turns (frames)")
    print(pd.crosstab(lr["group"], lr["side"]))
    print("\ndrive command in the same clips (frames)")
    print(pd.crosstab(lr["group"], lr["command"]))

    if len(sys.argv) > 2:
        c = pd.read_csv(sys.argv[2])
        c["group"] = c["clip"].map(group_of)
        c = c[c["group"].isin(["left", "right"])]
        print("\nCNN class in the same clips (frames)")
        print(pd.crosstab(c["group"], c["cnn_class"]))


if __name__ == "__main__":
    main()
