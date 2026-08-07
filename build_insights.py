"""Derive per-intersection recommendations from NYC's own crash record.

Three audiences, three time horizons, one dataset:

  EMS      seconds   where an ambulance is going, right now
  DOT      years     what to build so it stops happening
  counsel  ongoing   where the injuries actually are

Nothing here is invented. Interventions are keyed to the contributing factor
NYPD recorded on the report, using standard Vision Zero countermeasures.

    python build_insights.py     ->  writes insights.json
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

CRASH_API = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
RADIUS_M = 150

# NYPD contributing factor -> the countermeasure NYC DOT actually deploys for
# it. These are standard Vision Zero treatments, not invention.
INTERVENTION = {
    "Driver Inattention/Distraction": (
        "Daylighting + advance warning",
        "Remove parking within 20ft of the crosswalk so drivers see people "
        "waiting, and add an active warning beacon. This is the failure mode "
        "Ghost-V2X detects directly - a driver who has not registered a "
        "pedestrian already in the roadway."),
    "Failure to Yield Right-of-Way": (
        "Leading Pedestrian Interval",
        "Give pedestrians a 7-11 second head start before the parallel green. "
        "NYC has installed these widely and they are the standard treatment "
        "where turning drivers fail to yield."),
    "Turning Improperly": (
        "Protected turn phase or turn calming",
        "Separate the turn in time with a dedicated phase, or in space with "
        "rubber islands that force a slower turning radius."),
    "Pedestrian/Bicyclist/Other Pedestrian Error/Confusion": (
        "Crossing legibility",
        "High-visibility crosswalk markings, countdown timers, and refuge "
        "islands on wide crossings. Confusion is usually a design signal, not "
        "a behaviour problem."),
    "Unsafe Speed": (
        "Speed management",
        "Automated enforcement and geometric narrowing. Pedestrian survival "
        "falls sharply above 25mph, which is why NYC's default limit is 25."),
    "Traffic Control Disregarded": (
        "Signal visibility and enforcement",
        "Check signal head placement and sightlines before assuming wilful "
        "violation."),
}
DEFAULT_INTERVENTION = (
    "Study required",
    "The dominant contributing factor here has no single standard "
    "countermeasure. This intersection needs a site visit.")


def soda(**kw):
    url = CRASH_API + "?" + urllib.parse.urlencode(kw)
    req = urllib.request.Request(url, headers={"User-Agent": "ghost-v2x"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def profile(lat: float, lon: float) -> dict:
    rows = soda(**{
        "$select": "crash_time,contributing_factor_vehicle_1,"
                   "number_of_pedestrians_injured,number_of_cyclist_injured,"
                   "number_of_pedestrians_killed,number_of_cyclist_killed",
        "$where": f"within_circle(location,{lat},{lon},{RADIUS_M}) AND "
                  "(number_of_pedestrians_injured>0 OR number_of_cyclist_injured>0)",
        "$limit": 3000,
    })
    if not rows:
        return {}

    hours = Counter()
    for r in rows:
        t = r.get("crash_time")
        if t:
            hours[int(t.split(":")[0])] += 1

    factors = Counter(
        r.get("contributing_factor_vehicle_1") or "Unspecified" for r in rows)
    # "Unspecified" is a data-entry gap, not a cause; it cannot be treated.
    ranked = [(f, n) for f, n in factors.most_common() if f != "Unspecified"]
    top_factor = ranked[0][0] if ranked else "Unspecified"

    killed = sum(int(r.get("number_of_pedestrians_killed") or 0)
                 + int(r.get("number_of_cyclist_killed") or 0) for r in rows)

    name, detail = INTERVENTION.get(top_factor, DEFAULT_INTERVENTION)
    peak = hours.most_common(3)
    return {
        "crashes": len(rows),
        "killed": killed,
        "peak_hours": [{"hour": h, "count": n} for h, n in peak],
        "top_factors": [{"factor": f, "count": n} for f, n in ranked[:3]],
        "intervention": name,
        "rationale": detail,
    }


def main() -> None:
    ranked = json.loads(
        Path("camera_risk_ranking.json").read_text(encoding="utf-8"))
    out = []
    top = ranked[:20]
    print(f"profiling {len(top)} intersections against the crash record...\n")
    for i, cam in enumerate(top, 1):
        try:
            prof = profile(float(cam["latitude"]), float(cam["longitude"]))
        except Exception as exc:
            print(f"  {i:>2}. {cam['name'][:38]:<38} FAILED {exc}")
            continue
        if not prof:
            continue
        entry = {**cam, **prof}
        out.append(entry)
        peak = prof["peak_hours"][0]["hour"] if prof["peak_hours"] else "?"
        print(f"  {i:>2}. {cam['name'][:38]:<38} {cam['people_injured']:>3} hurt  "
              f"peak {peak:02d}:00  {prof['intervention']}")

    Path("insights.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote insights.json ({len(out)} intersections)")
    total_hurt = sum(e["people_injured"] for e in out)
    total_killed = sum(e["killed"] for e in out)
    print(f"across these 20 corners: {total_hurt} injured, {total_killed} killed")


if __name__ == "__main__":
    main()
