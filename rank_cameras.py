"""Rank NYCTMC cameras by documented pedestrian/cyclist harm nearby.

Choosing a camera by eye picks somewhere that *looks* busy. This picks where
people actually get hurt, using NYC's own crash record (Motor Vehicle
Collisions, Socrata h9gi-nx95).

That reframes the whole system: it is not "we watch a street", it is "we watch
the streets the city's own data says are hurting people, using cameras already
installed there."

    python rank_cameras.py                # top 15 by pedestrian/cyclist injury
    python rank_cameras.py --since 2020   # widen the window

One bulk query pulls every injury crash in the window, then cameras are matched
locally - 968 individual API calls would be slow and rude.
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

CRASH_API = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
CAMERA_LIST_URL = "https://webcams.nyctmc.org/api/cameras/"
RADIUS_M = 150

HIGHWAY_MARKERS = (
    "expy", "expwy", "pkwy", "belt", "bqe", "fdr", "thruway", "tunnel",
    "bridge", "br @", "-eb_", "-wb_", "-nb_", "-sb_", "ramp",
)


def get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "ghost-v2x"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def fetch_injury_crashes(since: str) -> list[tuple[float, float, int]]:
    """Every crash since `since` that injured a pedestrian or cyclist."""
    out, offset, page = [], 0, 50000
    while True:
        url = CRASH_API + "?" + urllib.parse.urlencode({
            "$select": "latitude,longitude,number_of_pedestrians_injured,"
                       "number_of_cyclist_injured",
            "$where": f"crash_date >= '{since}T00:00:00' AND latitude IS NOT NULL "
                      "AND (number_of_pedestrians_injured > 0 "
                      "OR number_of_cyclist_injured > 0)",
            "$limit": page, "$offset": offset,
        })
        rows = get_json(url)
        for r in rows:
            try:
                lat, lon = float(r["latitude"]), float(r["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            if lat == 0 or lon == 0:
                continue
            hurt = (int(r.get("number_of_pedestrians_injured") or 0)
                    + int(r.get("number_of_cyclist_injured") or 0))
            out.append((lat, lon, hurt))
        if len(rows) < page:
            break
        offset += page
    return out


def is_street_intersection(cam: dict) -> bool:
    name = str(cam.get("name", "")).lower()
    return "@" in name and not any(m in name for m in HIGHWAY_MARKERS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2021-01-01")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    print(f"fetching pedestrian/cyclist injury crashes since {args.since}...")
    crashes = fetch_injury_crashes(args.since)
    print(f"  {len(crashes)} crashes with injured pedestrians or cyclists\n")

    cams = get_json(CAMERA_LIST_URL)
    cams = cams if isinstance(cams, list) else cams.get("cameras", [])
    live = [c for c in cams
            if str(c.get("isOnline")).lower() == "true" and is_street_intersection(c)]
    print(f"scoring {len(live)} online street-intersection cameras...\n")

    # Degrees per metre varies with latitude; NYC is ~40.7N.
    deg_lat = RADIUS_M / 111_320.0
    deg_lon = RADIUS_M / (111_320.0 * math.cos(math.radians(40.7)))

    # Bucket crashes into a coarse grid so each camera checks only its neighbours.
    grid: dict[tuple[int, int], list] = {}
    for lat, lon, hurt in crashes:
        grid.setdefault((int(lat / deg_lat), int(lon / deg_lon)), []).append(
            (lat, lon, hurt))

    scored = []
    for cam in live:
        try:
            clat, clon = float(cam["latitude"]), float(cam["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        gy, gx = int(clat / deg_lat), int(clon / deg_lon)
        events = injured = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for lat, lon, hurt in grid.get((gy + dy, gx + dx), ()):
                    dlat = (lat - clat) / deg_lat
                    dlon = (lon - clon) / deg_lon
                    if dlat * dlat + dlon * dlon <= 1.0:
                        events += 1
                        injured += hurt
        if events:
            scored.append((injured, events, cam))

    scored.sort(key=lambda s: (-s[0], -s[1]))
    print(f"{'people hurt':>11} {'crashes':>8}  camera")
    print("-" * 78)
    for injured, events, cam in scored[: args.top]:
        print(f"{injured:>11} {events:>8}  {cam['name'][:44]:<44} {cam['area']}")
        print(f"{'':>21}  id={cam['id']}")

    if scored:
        Path("camera_risk_ranking.json").write_text(json.dumps([
            {"id": c["id"], "name": c["name"], "area": c["area"],
             "latitude": c["latitude"], "longitude": c["longitude"],
             "people_injured": i, "crashes": e}
            for i, e, c in scored[:100]
        ], indent=1), encoding="utf-8")
        print("\nwrote camera_risk_ranking.json (top 100)")


if __name__ == "__main__":
    main()
