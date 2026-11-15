#!/usr/bin/env python3
"""Rebuild the data of docs/replay.png and docs/label_rule.png, and the clip list, from check/results.

Writes into OUT_DIR:
  clips.dat        per lane clip: rank, share of frames with 2+ white pixels on both scan rows, frames, name
  turns.dat        turn clips: share of frames on the turn side, straight, other side, for the lane angle and the CNN;
                   straight is split into fallback (lane angle at 85) and forward (CNN class)
  check_vals.tex   share over all frames and the frame count
  sign.dat         rightsign_1: widest right-sign box per frame
  sign_sent.dat    rightsign_1: frames on which the rules select the right-sign label
and check/clips.csv (every replayed clip with its frame count).

usage: python check/make_figure_data.py OUT_DIR
then:  cd OUT_DIR && pdflatex replay.tex && pdflatex label_rule.tex
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    d = pd.read_csv(os.path.join(RES, "lane_track.csv"))
    c = pd.read_csv(os.path.join(RES, "cnn_track.csv"))
    det = pd.read_csv(os.path.join(RES, "detector_as_sent.csv"), keep_default_na=False)

    d["group"] = d["clip"].str.rsplit("_", n=1).str[0]
    c["group"] = c["clip"].str.rsplit("_", n=1).str[0]
    d["found"] = d["c_bot"].notna() & d["c_top"].notna()

    per = d.groupby("clip").agg(frames=("found", "size"), rate=("found", "mean")).reset_index()
    per = per.sort_values(["rate", "frames"], ascending=[False, False]).reset_index(drop=True)
    per["i"] = range(1, len(per) + 1)
    per["pct"] = 100 * per["rate"]
    per[["i", "pct", "frames", "clip"]].to_csv(os.path.join(out, "clips.dat"), sep=" ", index=False, float_format="%.3f")

    rows = []
    for g, turn, other in [("left", "left", "right"), ("right", "right", "left")]:
        x = d[d["group"] == g]
        side = pd.Series("straight", index=x.index)
        side[x["angle"] < 85.0] = "left"
        side[x["angle"] > 85.0] = "right"
        rows.append(dict(name=f"{g}_angle", n=len(x), turn=100 * (side == turn).mean(),
                         straight=100 * (side == "straight").mean(), other=100 * (side == other).mean()))
        y = c[c["group"] == g]
        cls = y["cnn_class"].str.lower()
        rows.append(dict(name=f"{g}_cnn", n=len(y), turn=100 * (cls == turn).mean(),
                         straight=100 * (cls == "forward").mean(), other=100 * (cls == other).mean()))
    t = pd.DataFrame(rows)
    t["y"] = [4, 3, 1.5, 0.5]
    is_cnn = t["name"].str.endswith("_cnn")
    t["fallback"] = t["straight"].where(~is_cnn, 0.0)   # lane angle at the fallback of 85
    t["forward"] = t["straight"].where(is_cnn, 0.0)     # CNN class forward
    t.to_csv(os.path.join(out, "turns.dat"), sep=" ", index=False, float_format="%.2f")

    with open(os.path.join(out, "check_vals.tex"), "w") as f:
        f.write("\\def\\allpct{%.1f}\n\\def\\nframes{%s}\n" % (100 * d["found"].mean(), f"{len(d):,}"))

    s = det[det["clip"] == "rightsign_1"].copy()

    def width(v):
        w = [int(x.split(":")[2]) for x in v.split() if x.startswith("rightsign:")]
        return max(w) if w else float("nan")
    s["w"] = s["detections"].map(width)
    s[["frame", "w"]].to_csv(os.path.join(out, "sign.dat"), sep=" ", index=False)
    s.loc[s["selected"] == "rightsign", ["frame"]].to_csv(os.path.join(out, "sign_sent.dat"), sep=" ", index=False)

    lane = d.groupby("clip").size().rename("frames").reset_index()
    lane["replay"] = "lane, cnn"
    obst = det.groupby("clip").size().rename("frames").reset_index()
    obst["replay"] = "detector"
    clips = pd.concat([lane, obst], ignore_index=True)
    clips.to_csv(os.path.join(HERE, "clips.csv"), index=False)
    print(t.round(1).to_string(index=False))
    print("lane clips", len(lane), int(lane["frames"].sum()), "detector clips", len(obst), int(obst["frames"].sum()))


if __name__ == "__main__":
    main()
