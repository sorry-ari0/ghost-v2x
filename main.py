"""Ghost-V2X - collision risk from cameras the city already owns.

Polls a live NYC DOT traffic camera, runs detection on each frame via
Roboflow's hosted API, projects detections onto the road plane, and
estimates closest-point-of-approach between vehicles and pedestrians.

Runs standalone: with no ROBOFLOW_API_KEY set it stays in FAIL_SAFE and
still serves, so the container is always deployable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ghost-v2x")

# --- configuration (all overridable without a redeploy) --------------------

CAMERA_LIST_URL = os.getenv("CAMERA_LIST_URL", "https://webcams.nyctmc.org/api/cameras/")

# Broadway @ 46 St - Quad South. Chosen by inspecting live frames from eight
# candidates: it is the only one combining heavy pedestrian volume, vehicle
# traffic, an elevated angle that actually shows the road surface, and a zebra
# crosswalk to calibrate against. Highway cameras are useless here - no
# pedestrians means no vehicle-pedestrian conflict to measure.
CAMERA_ID = os.getenv("CAMERA_ID", "1927b469-e2dc-4943-a70c-e6e52fd4c48c")
CAMERA_MATCH = os.getenv("CAMERA_MATCH", "Broadway @ 46 St- Quad South")
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "2.5"))

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL", "vehicle-detection-3mmwj/1")
ROBOFLOW_URL = os.getenv("ROBOFLOW_URL", "https://detect.roboflow.com")
# DOT frames are only 352x240, so a mid-ground pedestrian is ~25px tall. The
# usual 0.4 threshold discards most of them.
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))

# Closest-point-of-approach thresholds, in real-world units.
TTC_HORIZON_S = float(os.getenv("TTC_HORIZON_S", "8.0"))
MISS_DISTANCE_M = float(os.getenv("MISS_DISTANCE_M", "2.0"))

# Ground-plane calibration. Four points in the camera image mapped to four
# points in metres on the road surface. The defaults are a placeholder - see
# calibrate.py. Without a real calibration this is still image space, and two
# objects fifty feet apart can look like they are touching.
# Starting estimate for Broadway @ 46 St, read off a live frame: the open
# roadway carrying the crosswalk, mapped to roughly 15m across (4-5 lanes) by
# 25m deep. Refine it against scout/00_*.jpg - see calibrate.py. Being within
# 20% of true scale beats image space by a wide margin.
SRC_QUAD = os.getenv("SRC_QUAD", "0.32,0.48 0.88,0.45 1.00,0.90 0.20,0.95")
DST_QUAD = os.getenv("DST_QUAD", "0,25 15,25 15,0 0,0")

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle", "vehicle", "van"}
PEDESTRIAN_CLASSES = {"person", "pedestrian", "bicycle", "cyclist"}


# --- geometry --------------------------------------------------------------

def _parse_quad(raw: str) -> np.ndarray:
    pts = [tuple(float(v) for v in pair.split(",")) for pair in raw.split()]
    if len(pts) != 4:
        raise ValueError(f"quad needs exactly 4 points, got {len(pts)}: {raw!r}")
    return np.asarray(pts, dtype=float)


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve the 3x3 homography taking src quad to dst quad (DLT + SVD)."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    h = vt[-1].reshape(3, 3)
    return h / h[2, 2]


def _project(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = h @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-9:
        return float("inf"), float("inf")
    return float(p[0] / p[2]), float(p[1] / p[2])


def closest_approach(
    p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray,
) -> tuple[float, float]:
    """Return (seconds_to_closest_approach, distance_at_that_moment).

    Constant-velocity model. Negative time means they are already separating.
    """
    dp = p2 - p1
    dv = v2 - v1
    closing = float(dv @ dv)
    if closing < 1e-9:
        return float("inf"), float(np.linalg.norm(dp))
    t = -float(dp @ dv) / closing
    miss = float(np.linalg.norm(dp + dv * t))
    return t, miss


# --- tracking --------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    kind: str
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_seen: float = 0.0
    hits: int = 1


# Plausible top speeds, used to size the association gate. At a 2.5s cadence a
# car covers 20-30m between frames, so a fixed small gate rejects every real
# vehicle match and every car becomes a new track with no velocity history.
MAX_SPEED_MPS = {"vehicle": 25.0, "pedestrian": 3.5}


class Tracker:
    """Greedy association in ground-plane metres, gated on predicted position."""

    def __init__(self, ttl_s: float = 6.0) -> None:
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.ttl_s = ttl_s

    def update(self, dets: list[tuple[str, np.ndarray]], now: float) -> list[Track]:
        unmatched = set(self.tracks)
        for kind, pos in dets:
            best_id, best_d = None, None
            for tid in unmatched:
                track = self.tracks[tid]
                if track.kind != kind:
                    continue
                dt = max(now - track.last_seen, 1e-3)
                reach = MAX_SPEED_MPS.get(kind, 25.0) * dt
                if track.hits > 1:
                    # Once a track has velocity, match against where it should
                    # be and gate tightly on the residual.
                    reference = track.pos + track.vel * dt
                    gate = max(4.0, reach * 0.5)
                else:
                    reference = track.pos
                    gate = reach
                d = float(np.linalg.norm(reference - pos))
                if d <= gate and (best_d is None or d < best_d):
                    best_id, best_d = tid, d

            if best_id is None:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, kind, pos, last_seen=now)
                continue

            track = self.tracks[best_id]
            dt = max(now - track.last_seen, 1e-3)
            measured = (pos - track.pos) / dt
            # Smooth hard, because a 2.5s baseline makes single-frame velocity
            # extremely noisy - one bounding-box wobble becomes a phantom 5 m/s.
            track.vel = 0.6 * track.vel + 0.4 * measured if track.hits > 1 else measured
            track.pos = pos
            track.last_seen = now
            track.hits += 1
            unmatched.discard(best_id)

        for tid in [t for t, tr in self.tracks.items() if now - tr.last_seen > self.ttl_s]:
            del self.tracks[tid]
        return list(self.tracks.values())


# --- shared state ----------------------------------------------------------

@dataclass
class State:
    status: str = "STARTING"
    reason: str = "not yet polled"
    camera: dict[str, Any] = field(default_factory=dict)
    risk: str = "UNKNOWN"
    conflicts: list[dict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    frames: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    last_ok: float | None = None
    updated: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        age = None if self.last_ok is None else round(time.time() - self.last_ok, 1)
        return {
            "status": self.status,
            "reason": self.reason,
            "risk": self.risk,
            "conflicts": self.conflicts,
            "counts": self.counts,
            "camera": self.camera,
            "frames": self.frames,
            "errors": self.errors,
            "seconds_since_good_frame": age,
            "updated": self.updated,
        }


STATE = State()
TRACKER = Tracker()


# --- pipeline --------------------------------------------------------------

def _bundled_cameras() -> list[dict]:
    """The roster snapshot committed alongside the app.

    Camera IDs are stable, so a stale roster still resolves to a working image
    URL. This exists so an outage of the *list* endpoint cannot take the demo
    down when the *image* endpoints are perfectly healthy.
    """
    path = Path(__file__).parent / "cameras.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


async def pick_camera(client: httpx.AsyncClient) -> dict:
    try:
        r = await client.get(CAMERA_LIST_URL, timeout=15)
        r.raise_for_status()
        payload = r.json()
        cams = payload if isinstance(payload, list) else payload.get("cameras", [])
    except Exception as exc:
        cams = _bundled_cameras()
        if not cams:
            raise
        log.warning("camera list unavailable (%s); using bundled roster", exc)

    def online(c: dict) -> bool:
        v = c.get("isOnline", c.get("online", True))
        return str(v).lower() in ("true", "1", "yes")

    live = [c for c in cams if online(c)]
    if not live:
        raise RuntimeError(f"no online cameras in {len(cams)} returned")

    if CAMERA_ID:
        for c in live:
            if str(c.get("id", c.get("cameraId", ""))) == CAMERA_ID:
                return c
    needle = CAMERA_MATCH.lower()
    for c in live:
        if needle in str(c.get("name", "")).lower():
            return c
    log.warning("no camera matched %r; falling back to first online", CAMERA_MATCH)
    return live[0]


def camera_image_url(cam: dict) -> str:
    for key in ("imageUrl", "image_url", "url"):
        if cam.get(key):
            return str(cam[key])
    cid = cam.get("id") or cam.get("cameraId")
    return f"https://webcams.nyctmc.org/api/cameras/{cid}/image"


async def detect(client: httpx.AsyncClient, jpeg: bytes) -> list[dict]:
    r = await client.post(
        f"{ROBOFLOW_URL}/{ROBOFLOW_MODEL}",
        params={"api_key": ROBOFLOW_API_KEY, "confidence": CONFIDENCE},
        content=jpeg,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("predictions", [])


def classify(label: str) -> str | None:
    low = label.lower()
    if low in VEHICLE_CLASSES:
        return "vehicle"
    if low in PEDESTRIAN_CLASSES:
        return "pedestrian"
    return None


def ground_positions(preds: list[dict], w: float, h: float, h_mat: np.ndarray):
    out = []
    for p in preds:
        kind = classify(str(p.get("class", "")))
        if kind is None:
            continue
        # Bottom-centre of the box is where the object meets the road. The box
        # centre floats in the air and projects to nonsense on the ground plane.
        cx = float(p["x"]) / w
        cy = (float(p["y"]) + float(p["height"]) / 2.0) / h
        out.append((kind, np.array(_project(h_mat, cx, cy))))
    return out


def assess(tracks: list[Track]) -> tuple[str, list[dict]]:
    vehicles = [t for t in tracks if t.kind == "vehicle" and t.hits >= 2]
    peds = [t for t in tracks if t.kind == "pedestrian" and t.hits >= 2]

    conflicts = []
    for v in vehicles:
        for p in peds:
            t, miss = closest_approach(v.pos, v.vel, p.pos, p.vel)
            if 0 < t <= TTC_HORIZON_S and miss <= MISS_DISTANCE_M:
                conflicts.append({
                    "vehicle_track": v.track_id,
                    "pedestrian_track": p.track_id,
                    "seconds_to_closest_approach": round(t, 2),
                    "miss_distance_m": round(miss, 2),
                })

    conflicts.sort(key=lambda c: c["seconds_to_closest_approach"])
    if not conflicts:
        return "CLEAR", []
    soonest = conflicts[0]["seconds_to_closest_approach"]
    level = "HIGH" if soonest <= 3.0 else "MEDIUM" if soonest <= 5.0 else "LOW"
    return level, conflicts[:5]


async def loop() -> None:
    """Poll → detect → track → assess. Every failure path degrades, never raises."""
    h_mat = _homography(_parse_quad(SRC_QUAD), _parse_quad(DST_QUAD))
    cam: dict = {}

    async with httpx.AsyncClient(follow_redirects=True) as client:
        while True:
            try:
                if not cam:
                    cam = await pick_camera(client)
                    STATE.camera = {
                        "id": cam.get("id") or cam.get("cameraId"),
                        "name": cam.get("name"),
                        "lat": cam.get("latitude"),
                        "lon": cam.get("longitude"),
                    }
                    log.info("camera locked: %s", STATE.camera.get("name"))

                # Cache-bust; DOT still images sit behind a CDN.
                url = camera_image_url(cam)
                sep = "&" if "?" in url else "?"
                r = await client.get(f"{url}{sep}t={int(time.time() * 1000)}", timeout=20)
                r.raise_for_status()
                jpeg = r.content
                if len(jpeg) < 1024:
                    raise RuntimeError(f"frame too small ({len(jpeg)}B) - camera likely dark")

                if not ROBOFLOW_API_KEY:
                    STATE.status = "FAIL_SAFE"
                    STATE.reason = "ROBOFLOW_API_KEY not set - detection disabled"
                    STATE.risk = "UNKNOWN"
                    STATE.frames += 1
                    STATE.updated = time.time()
                    await asyncio.sleep(POLL_SECONDS)
                    continue

                preds = await detect(client, jpeg)
                iw = float(preds[0].get("image_width", 1280)) if preds else 1280.0
                ih = float(preds[0].get("image_height", 720)) if preds else 720.0
                dets = ground_positions(preds, iw, ih, h_mat)

                now = time.time()
                tracks = TRACKER.update(dets, now)
                risk, conflicts = assess(tracks)

                STATE.status = "ACTIVE"
                STATE.reason = "ok"
                STATE.risk = risk
                STATE.conflicts = conflicts
                STATE.counts = {
                    "vehicles": sum(1 for t in tracks if t.kind == "vehicle"),
                    "pedestrians": sum(1 for t in tracks if t.kind == "pedestrian"),
                    "detections": len(dets),
                }
                STATE.frames += 1
                STATE.last_ok = now
                STATE.consecutive_errors = 0
                STATE.updated = now

            except Exception as exc:
                STATE.errors += 1
                STATE.consecutive_errors += 1
                STATE.updated = time.time()
                log.warning("cycle failed (%d in a row): %s", STATE.consecutive_errors, exc)

                # Two strikes and we stop asserting anything about the street.
                # A stale HIGH is far more dangerous than an honest UNKNOWN.
                if STATE.consecutive_errors >= 2:
                    STATE.status = "FAIL_SAFE"
                    STATE.reason = f"{type(exc).__name__}: {exc}"
                    STATE.risk = "UNKNOWN"
                    STATE.conflicts = []
                    TRACKER.tracks.clear()
                if STATE.consecutive_errors >= 5:
                    cam = {}  # re-select; this one may be down for maintenance

            await asyncio.sleep(POLL_SECONDS)


# --- web -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(title="Ghost-V2X", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    """Liveness only. The pipeline degrades to FAIL_SAFE without the container
    being unhealthy - Cloud Run must not restart us for a dark camera."""
    return {"ok": True}


@app.get("/api/state")
def api_state():
    return JSONResponse(STATE.as_dict())


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!doctype html><html><head><meta charset="utf-8">
<title>Ghost-V2X</title><style>
:root{color-scheme:dark}
body{margin:0;font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
background:#0b0e14;color:#c8d0e0;padding:2rem}
h1{font-size:1.1rem;letter-spacing:.14em;text-transform:uppercase;color:#7d8799;margin:0 0 .25rem}
.sub{color:#5a6373;margin-bottom:2rem}
.risk{font-size:3.5rem;font-weight:700;letter-spacing:-.02em;margin:.2rem 0}
.CLEAR{color:#3ddc97}.LOW{color:#ffd166}.MEDIUM{color:#ff9f45}.HIGH{color:#ff4d5e}
.UNKNOWN{color:#5a6373}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:2rem 0}
.card{background:#131823;border:1px solid #1f2634;border-radius:10px;padding:1rem}
.k{color:#5a6373;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase}
.v{font-size:1.5rem;margin-top:.35rem}
table{width:100%;border-collapse:collapse;margin-top:.5rem}
th,td{text-align:left;padding:.5rem;border-bottom:1px solid #1f2634}
th{color:#5a6373;font-size:.72rem;letter-spacing:.08em;text-transform:uppercase}
.pill{display:inline-block;padding:.2rem .6rem;border-radius:99px;font-size:.75rem;
border:1px solid #2a3346}
</style></head><body>
<h1>Ghost-V2X</h1>
<div class="sub">Collision risk from cameras the city already owns.</div>
<div id="cam" class="pill">connecting…</div>
<div class="risk UNKNOWN" id="risk">-</div>
<div class="sub" id="reason"></div>
<div class="grid">
  <div class="card"><div class="k">Vehicles</div><div class="v" id="veh">-</div></div>
  <div class="card"><div class="k">Pedestrians</div><div class="v" id="ped">-</div></div>
  <div class="card"><div class="k">Frames</div><div class="v" id="frames">-</div></div>
  <div class="card"><div class="k">Frame age</div><div class="v" id="age">-</div></div>
</div>
<table><thead><tr><th>Vehicle</th><th>Pedestrian</th><th>Closest approach</th>
<th>Miss distance</th></tr></thead><tbody id="rows">
<tr><td colspan="4" style="color:#5a6373">no conflicts</td></tr></tbody></table>
<script>
async function tick(){
  try{
    const s = await (await fetch('/api/state')).json();
    const r = document.getElementById('risk');
    r.textContent = s.risk; r.className = 'risk ' + s.risk;
    document.getElementById('reason').textContent =
      s.status === 'ACTIVE' ? '' : s.status + ' - ' + s.reason;
    document.getElementById('cam').textContent =
      (s.camera && s.camera.name) ? s.camera.name : 'no camera';
    document.getElementById('veh').textContent = s.counts.vehicles ?? '-';
    document.getElementById('ped').textContent = s.counts.pedestrians ?? '-';
    document.getElementById('frames').textContent = s.frames;
    document.getElementById('age').textContent =
      s.seconds_since_good_frame == null ? '-' : s.seconds_since_good_frame + 's';
    const rows = document.getElementById('rows');
    rows.innerHTML = s.conflicts.length ? s.conflicts.map(c =>
      `<tr><td>#${c.vehicle_track}</td><td>#${c.pedestrian_track}</td>
       <td>${c.seconds_to_closest_approach}s</td><td>${c.miss_distance_m}m</td></tr>`
    ).join('') : '<tr><td colspan="4" style="color:#5a6373">no conflicts</td></tr>';
  }catch(e){ /* keep the last good frame on screen */ }
}
tick(); setInterval(tick, 1500);
</script></body></html>"""
