"""Grab one frame from the chosen camera and help you pick the ground quad.

Bounding boxes live in image space. Two objects whose boxes nearly touch on
screen can be fifty feet apart in reality - a pedestrian on the near sidewalk
and a car across the intersection will read as an imminent collision all night.
Mapping the road surface to metres is what makes the physics mean anything.

    python calibrate.py                 # saves frame.jpg, prints the defaults
    python calibrate.py --show          # opens it if you have a viewer

Then eyeball four points on the ROAD SURFACE in frame.jpg, in this order:
    far-left, far-right, near-right, near-left
Express each as a fraction of width,height (0-1). Estimate the real distances
in metres between them (a NYC traffic lane is ~3.0-3.7m wide; a crosswalk is
typically 2.4-4.6m deep). Then set:

    SRC_QUAD="0.21,0.52 0.79,0.54 1.02,0.98 -0.03,0.97"
    DST_QUAD="0,24 14,24 14,0 0,0"

Rough is fine. Being within 20% of true scale beats image space by a mile.
"""
from __future__ import annotations

import argparse
import asyncio
import os

import httpx

from main import CAMERA_MATCH, camera_image_url, pick_camera


async def grab(path: str) -> None:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        cam = await pick_camera(client)
        print(f"camera : {cam.get('name')}  (id={cam.get('id') or cam.get('cameraId')})")
        r = await client.get(camera_image_url(cam), timeout=20)
        r.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(r.content)
        print(f"saved  : {path}  ({len(r.content) / 1024:.0f} KB)")
        print(f"\nmatched on CAMERA_MATCH={CAMERA_MATCH!r}")
        print("pin this camera with:  CAMERA_ID=" +
              str(cam.get("id") or cam.get("cameraId")))
        print("\ncurrent quads (placeholders - replace them):")
        print(f'  SRC_QUAD="{os.getenv("SRC_QUAD", "0.15,0.55 0.85,0.55 1.05,1.0 -0.05,1.0")}"')
        print(f'  DST_QUAD="{os.getenv("DST_QUAD", "0,30 20,30 20,0 0,0")}"')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="frame.jpg")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    asyncio.run(grab(args.out))

    if args.show:
        import subprocess
        import sys
        opener = {"win32": ["cmd", "/c", "start", ""],
                  "darwin": ["open"]}.get(sys.platform, ["xdg-open"])
        subprocess.run(opener + [args.out], check=False)
