"""Validate the camera ranking against NYC's own Vision Zero designation.

rank_cameras.py scores cameras by crash history. That is our method. The
question is whether the method is any good, and NYC answers it for us: DOT
publishes its own list of Vision Zero Priority Intersections (tmt9-43em),
derived independently through their own analysis.

If our ranking reproduces their list without ever having seen it, the method
is validated by something other than our own reasoning.

    python validate_ranking.py
"""
from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request

HW = ("expy", "expwy", "pkwy", "belt", "bqe", "fdr", "thruway", "tunnel",
      "bridge", "br @", "-eb_", "-wb_", "-nb_", "-sb_", "ramp")


def soda(dataset: str, **kw):
    url = f"https://data.cityofnewyork.us/resource/{dataset}.json?" + \
        urllib.parse.urlencode(kw)
    req = urllib.request.Request(url, headers={"User-Agent": "ghost-v2x"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def metres(alat, alon, blat, blon) -> float:
    return math.hypot((alat - blat) * 111_320,
                      (alon - blon) * 111_320 * math.cos(math.radians(40.7)))


def main() -> None:
    priority = soda("tmt9-43em", **{"$limit": 5000})
    ranked = json.load(open("camera_risk_ranking.json", encoding="utf-8"))
    cameras = json.load(open("cameras.json", encoding="utf-8"))

    usable = []
    for c in cameras:
        name = str(c.get("name", "")).lower()
        if str(c.get("isOnline")).lower() != "true" or "@" not in name:
            continue
        if any(m in name for m in HW):
            continue
        try:
            usable.append((float(c["latitude"]), float(c["longitude"]), c))
        except (KeyError, TypeError, ValueError):
            continue

    pts = []
    for p in priority:
        try:
            pts.append((float(p["lat"]), float(p["long"]), p))
        except (KeyError, TypeError, ValueError):
            continue

    print("Agreement: our crash-ranked top 12 vs NYC's priority list")
    hits = 0
    for c in ranked[:12]:
        clat, clon = float(c["latitude"]), float(c["longitude"])
        match = any(metres(clat, clon, la, lo) < 200 for la, lo, _ in pts)
        hits += match
        print(f"  {'MATCH' if match else '  -  '}  {c['people_injured']:>3} hurt  "
              f"{c['name'][:46]}")
    print(f"\n  {hits}/12 agree\n")

    covered = sum(
        1 for la, lo, _ in pts
        if any(metres(la, lo, cla, clo) < 200 for cla, clo, _ in usable))
    total = len(pts)
    print(f"Deployment coverage")
    print(f"  NYC Vision Zero Priority Intersections : {total}")
    print(f"  already watched by a usable camera     : {covered} "
          f"({covered * 100 // total}%)")
    print(f"  no camera within 200m                  : {total - covered}")
    print(f"\n  Ghost-V2X is deployable to {covered} of the city's own priority")
    print("  intersections today, with zero new hardware.")


if __name__ == "__main__":
    main()
