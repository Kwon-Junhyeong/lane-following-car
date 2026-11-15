#!/usr/bin/env python3
"""Summarize the CSV written by replay_detector.py, per group of clips.

usage: python check/summarize_detector.py detector.csv
"""
import sys

import pandas as pd

EXPECTED = {
    "gateclosed": "gateclosed", "closed": "gateclosed", "gateopen": "gateopen", "open": "gateopen",
    "hill": "smallhill", "leftsign": "leftsign", "redlight": "redlight", "rightsign": "rightsign",
    "staticb": "staticb", "staticy": "staticy",
}


def main():
    d = pd.read_csv(sys.argv[1], keep_default_na=False)
    d["group"] = d["clip"].str.rsplit("_", n=1).str[0]
    rows = []
    for (group, clip), c in d.groupby(["group", "clip"], sort=True):
        exp = EXPECTED.get(group)
        seen = c["detections"].str.contains(f"{exp}:") if exp else pd.Series(False, index=c.index)
        sel = c["selected"] == exp if exp else pd.Series(False, index=c.index)
        other = (c["selected"] != "") & ~sel
        first_seen = int(c.loc[seen, "frame"].min()) if seen.any() else -1
        first_sel = int(c.loc[sel, "frame"].min()) if sel.any() else -1
        rows.append(dict(group=group, clip=clip, frames=len(c), detected=int(seen.sum()),
                         selected=int(sel.sum()), first_detected=first_seen,
                         first_selected=first_sel, other_label=int(other.sum()),
                         other_labels=" ".join(sorted(set(c.loc[other, "selected"])))))
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(r.to_string(index=False))
    r["object"] = r["group"].map(lambda g: EXPECTED.get(g, "(none: " + g + ")"))
    g = r.groupby("object").agg(
        clips=("clip", "size"), frames=("frames", "sum"), detected=("detected", "sum"),
        selected=("selected", "sum"), other_label=("other_label", "sum"),
        first_detected_min=("first_detected", "min"), first_detected_max=("first_detected", "max"),
        first_selected_min=("first_selected", lambda s: int(s[s >= 0].min()) if (s >= 0).any() else -1),
        first_selected_max=("first_selected", "max"),
        not_selected=("first_selected", lambda s: int((s < 0).sum())))
    print()
    print(g.to_string())
    print("\nclips", len(r), "frames", int(r["frames"].sum()))


if __name__ == "__main__":
    main()
