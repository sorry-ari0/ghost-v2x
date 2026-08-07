"""Score cameras for demo suitability and pull candidate frames to eyeball.

A highway camera is useless here: no pedestrians means no vehicle-pedestrian
conflict, so TTC has nothing to measure. What this needs is a signalised
intersection with crosswalks, shot from enough height to see the road surface.

    python scout_cameras.py            # save list, fetch top candidates
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

CAMERA_LIST_URL = "https://webcams.nyctmc.org/api/cameras/"
OUT_DIR = Path("scout")

# Signalised crossings in dense pedestrian areas. Times Square and the Lower
# East Side crossings carry heavy foot traffic all day.
PREFERRED = [
    "Broadway @ 46 St- Quad South",
    "Broadway @ 42 St",
    "Delancy St @ Essex St",
    "Canal Street @ Chrystie Street",
    "Flatbush Ave @ Tillary St",
    "1 Ave @ 42 St",
    "Union Sq @ 14 St",
    "Houston St @ Broadway",
]

# Substrings that mark a limited-access road: no pedestrians, so no conflicts.
HIGHWAY_MARKERS = (
    "expy", "expwy", "pkwy", "belt", "bqe", "fdr", "thruway", "tunnel",
    "bridge", "br @", "-eb_", "-wb_", "-nb_", "-sb_", "ramp",
)


def is_street_intersection(cam: dict) -> bool:
    name = str(cam.get("name", "")).lower()
    if any(marker in name for marker in HIGHWAY_MARKERS):
        return False
    return "@" in name


async def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    async with httpx.AsyncClient(follow_redirects=True) as client:
        r = await client.get(CAMERA_LIST_URL, timeout=20)
        r.raise_for_status()
        cams = r.json()
        cams = cams if isinstance(cams, list) else cams.get("cameras", [])

        # Bundle the roster so a dead list endpoint cannot stop the demo.
        Path("cameras.json").write_text(
            json.dumps(cams, indent=1), encoding="utf-8")

        online = [c for c in cams if str(c.get("isOnline")).lower() == "true"]
        streets = [c for c in online if is_street_intersection(c)]
        print(f"{len(cams)} cameras | {len(online)} online | "
              f"{len(streets)} street intersections")

        by_name = {str(c.get("name", "")).strip(): c for c in online}
        picks = []
        for want in PREFERRED:
            match = next(
                (c for n, c in by_name.items() if want.lower() in n.lower()), None)
            if match:
                picks.append(match)

        print(f"\nfetching {len(picks)} candidate frames -> {OUT_DIR}/\n")
        for index, cam in enumerate(picks):
            slug = f"{index:02d}_" + "".join(
                ch if ch.isalnum() else "_" for ch in str(cam["name"])[:40])
            try:
                img = await client.get(cam["imageUrl"], timeout=20)
                img.raise_for_status()
                path = OUT_DIR / f"{slug}.jpg"
                path.write_bytes(img.content)
                print(f"  {path}  ({len(img.content) // 1024} KB)  "
                      f"{cam['name']}  id={cam['id']}")
            except Exception as exc:
                print(f"  FAILED {cam['name']}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
