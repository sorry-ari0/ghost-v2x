"""Ghost-V2X - collision risk from cameras the city already owns.

Polls a live NYC DOT traffic camera, runs detection on each frame via
Roboflow's hosted API, projects detections onto the road plane, and
estimates closest-point-of-approach between vehicles and pedestrians.

Runs standalone: with no ROBOFLOW_API_KEY set it stays in FAIL_SAFE and
still serves, so the container is always deployable.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
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

# Lenox Ave @ 125 St, Harlem. Ranked #2 of 619 street-intersection cameras by
# pedestrians and cyclists injured within 150m since 2021 (98 people, 95
# crashes) using NYC's own collision record - see rank_cameras.py.
#
# Rank alone is not enough, so every top candidate was inspected. #1 Delancy @
# Essex is a foreshortened view down the roadway with almost no pedestrians
# visible. #5 Broadway @ 43 St is the Times Square pedestrian plaza: huge foot
# traffic, essentially no vehicles, so its crashes happen outside the frame.
# Lenox is the highest-ranked camera that actually shows the conflict it is
# scored on - pedestrians crossing perpendicular to vehicle flow, over a zebra
# crosswalk that doubles as the calibration reference.
CAMERA_ID = os.getenv("CAMERA_ID", "156b0613-239a-4e77-aa0e-0a4becfc0b05")
CAMERA_MATCH = os.getenv("CAMERA_MATCH", "Lenox Ave @ 125 St")

# Measured: this camera publishes a new frame every ~2.0s (min 1.2, max 2.5).
# Poll faster than that and dedupe, so a new frame is picked up promptly
# instead of being missed by an aliased 2.5s cycle. Duplicate frames are
# discarded before inference - see the hash check in loop().
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.0"))

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
# Must detect BOTH vehicles and people in one pass. Most "vehicle detection"
# models on Universe have no person class at all, which silently yields zero
# pedestrians and therefore zero conflicts - the system looks healthy and
# reports CLEAR forever. A COCO-trained model covers person, car, truck, bus,
# bicycle, and motorcycle together.
# coco/38 verified against a live Lenox frame: 27 detections, 5 people, 14
# vehicles. coco/9 was the previous default and returns 1 car and ZERO people
# on the same frame - which yields no vehicle-pedestrian conflicts ever, while
# the system looks perfectly healthy. Do not change this without re-running
# probe_model.py.
ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL", "coco/38")
ROBOFLOW_URL = os.getenv("ROBOFLOW_URL", "https://detect.roboflow.com")
# DOT frames are only 352x240, so a mid-ground pedestrian is ~25px tall. The
# usual 0.4 threshold discards most of them.
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))
ROBOFLOW_OVERLAP = int(os.getenv("ROBOFLOW_OVERLAP", "45"))
# Upscale before inference. Measured on a live Lenox frame: the raw 352x240
# image returns ZERO detections, while the same frame at 704x480 returns 40+
# including 20+ pedestrians. Detectors are trained near 640x640, and a 25px
# pedestrian falls below what they resolve. Without this the system reports
# CLEAR forever while appearing perfectly healthy - the worst failure mode
# there is, because nothing looks broken.
#
# Coordinate-safe: Roboflow echoes image_width/image_height for whatever it
# received, and ground_positions() divides by those, so the normalised
# fractions the homography consumes are identical either way.
UPSCALE = float(os.getenv("UPSCALE", "2.0"))

# Closest-point-of-approach thresholds, in real-world units.
TTC_HORIZON_S = float(os.getenv("TTC_HORIZON_S", "8.0"))
MISS_DISTANCE_M = float(os.getenv("MISS_DISTANCE_M", "2.0"))

# Ground-plane calibration. Four points in the camera image mapped to four
# points in metres on the road surface. The defaults are a placeholder - see
# calibrate.py. Without a real calibration this is still image space, and two
# objects fifty feet apart can look like they are touching.
# Starting estimate for Lenox Ave @ 125 St, read off a live frame: the roadway
# carrying the foreground crosswalk, mapped to roughly 18m across (Lenox is a
# wide avenue) by 22m deep. This is an eyeball estimate and the single highest
# leverage thing to refine - use the crosswalk as a ruler, see calibrate.py.
SRC_QUAD = os.getenv("SRC_QUAD", "0.28,0.60 0.66,0.56 0.78,0.97 0.05,0.93")
# Scale corrected from a physical measurement, not an estimate. With the
# quad set to 18x22m, tracked pedestrians who were actively crossing showed a
# p90 speed of 0.39 m/s. Humans walk at ~1.4 m/s, so the ground plane was
# 1.4/0.39 = 3.6x too small, and every distance and TTC was wrong by that
# factor. A camera looking down an avenue does see this far - roughly 65m
# across and 79m deep is consistent with the view.
#
# Check it yourself: /api/state reports scale_error_vs_walking. A value near
# 1.0 means the calibration is sound.
DST_QUAD = os.getenv("DST_QUAD", "0,79 65,79 65,0 0,0")

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

# A track whose implied speed exceeds this is not a real object moving fast,
# it is an identity switch. At 15 pedestrians in frame, greedy association
# occasionally hands track A the detection belonging to track B; the apparent
# jump becomes a phantom velocity aimed at whatever is nearby, and that is
# what produced sub-second TTCs against stationary traffic.
IMPLAUSIBLE_SPEED_MPS = {"vehicle": 32.0, "pedestrian": 6.0}

# Two observations is exactly what a velocity requires, and no more should be
# demanded. Raising this to 3 costs a full cycle - measured at a 2s cadence,
# the alertable moment (CPA 2.0s out, 0.0m miss) occurs at hits=2, and by
# hits=3 the encounter has already passed. Whether a velocity is TRUSTWORTHY
# is answered by the plausibility gate below, which tests physics rather than
# imposing an arbitrary delay.
MIN_HITS_FOR_CONFLICT = 2

# A marginal conflict must survive consecutive assessments before it is
# reported; one-frame coincidences are the dominant false alarm at a busy
# intersection. This deliberately does NOT apply to an imminent conflict:
# waiting a cycle to confirm a 2-second TTC spends most of the warning. The
# same principle as the alert throttle - filter the marginal, never delay the
# urgent.
CONFLICT_PERSISTENCE = 2
IMMINENT_TTC_S = 3.0


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
    duplicate_frames: int = 0
    median_ped_speed: float | None = None
    near_misses: list = field(default_factory=list)
    alerts_sent: int = 0
    alert_error: str = ""
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
            "duplicate_frames": self.duplicate_frames,
            "near_miss_count": len(self.near_misses),
            "median_ped_speed_mps": self.median_ped_speed,
            "scale_error_vs_walking": (
                round(self.median_ped_speed / 1.4, 2)
                if self.median_ped_speed else None),
            "alerts_sent": self.alerts_sent,
            "alert_error": self.alert_error,
            "errors": self.errors,
            "seconds_since_good_frame": age,
            "updated": self.updated,
        }


def _scale_error_value(self):
    return (round(self.median_ped_speed / 1.4, 2)
            if self.median_ped_speed else None)


State.scale_error_value = _scale_error_value

CONFLICT_LOG = Path("/tmp/ghost_v2x_conflicts.json")


def record_conflict(camera_id, camera_name, risk, ttc, miss_m, at) -> None:
    """Append to the durable per-location conflict record.

    This is the Traffic Conflict Technique, automated. Counting near-misses
    assesses an intersection without waiting years for crashes to accumulate,
    which is established road-safety practice; it is rarely done only because
    it has meant a human observer at the corner with a clipboard for days.

    Crashes are lagging and rare. Conflicts are leading and frequent. A corner
    generating conflicts but not yet crashes is where the next one happens,
    and that is the layer the planning map is missing.
    """
    try:
        log = (json.loads(CONFLICT_LOG.read_text(encoding="utf-8"))
               if CONFLICT_LOG.exists() else {})
    except Exception:
        log = {}
    entry = log.setdefault(
        camera_id, {"camera_name": camera_name, "conflicts": [], "first_seen": at})
    entry["camera_name"] = camera_name
    entry["conflicts"].append(
        {"at": round(at, 1), "risk": risk, "ttc": ttc, "miss_m": miss_m})
    entry["conflicts"] = entry["conflicts"][-500:]   # a rate signal, not an archive
    entry["last_seen"] = at
    try:
        CONFLICT_LOG.write_text(json.dumps(log), encoding="utf-8")
    except Exception as exc:
        log.warning if False else None
        logging.getLogger("ghost-v2x").warning("could not persist conflict: %s", exc)


def conflict_summary() -> dict:
    """Per-location conflict counts and observed hours, for the planning map."""
    try:
        raw = (json.loads(CONFLICT_LOG.read_text(encoding="utf-8"))
               if CONFLICT_LOG.exists() else {})
    except Exception:
        return {}
    out = {}
    for cam_id, e in raw.items():
        cs = e.get("conflicts", [])
        if not cs:
            continue
        hours = max((e.get("last_seen", 0) - e.get("first_seen", 0)) / 3600.0, 1 / 60)
        out[cam_id] = {
            "camera_name": e.get("camera_name"),
            "conflicts": len(cs),
            "high": sum(1 for c in cs if c.get("risk") == "HIGH"),
            "observed_hours": round(hours, 2),
            "per_hour": round(len(cs) / hours, 1),
            "worst_ttc": min((c["ttc"] for c in cs if c.get("ttc") is not None),
                             default=None),
        }
    return out


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


def upscale_jpeg(jpeg: bytes, factor: float) -> bytes:
    """Enlarge the frame before inference. Returns the original on any failure."""
    if factor <= 1.0:
        return jpeg
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(jpeg))
        big = img.resize((round(img.width * factor), round(img.height * factor)),
                         Image.LANCZOS)
        buf = io.BytesIO()
        big.convert("RGB").save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    except Exception as exc:
        # Never let a resize failure stop detection; degraded recall beats none.
        log.warning("upscale failed, sending original frame: %s", exc)
        return jpeg


def jpeg_dimensions(jpeg: bytes) -> tuple[int, int]:
    """Dimensions of the frame we are actually sending.

    Deliberately measured locally rather than read back from the inference
    response. Roboflow reports image size at the top level of the response,
    not per-prediction, so reading it off a prediction silently yields a
    fallback - and dividing 704-space coordinates by a wrong width puts every
    object in the wrong place on the ground plane with no error raised. The
    TTC numbers stay plausible and are entirely false.
    """
    from PIL import Image
    import io
    try:
        with Image.open(io.BytesIO(jpeg)) as img:
            return img.width, img.height
    except Exception as exc:
        # An undecodable frame is a real degradation, but the raw PIL error
        # ("cannot identify image file <_io.BytesIO object at 0x...>") tells an
        # operator nothing about which camera or why.
        raise RuntimeError(
            f"camera frame is not a decodable image ({len(jpeg)} bytes)"
        ) from exc


async def detect(client: httpx.AsyncClient, jpeg: bytes) -> list[dict]:
    """Run hosted inference on one frame.

    detect.roboflow.com wants the image base64-encoded in the request body
    with a form content-type - posting raw JPEG bytes fails. Its `confidence`
    query parameter is a percentage (0-100), not a 0-1 fraction; sending 0.25
    there reads as 0%, which returns every low-confidence box in the frame.
    """
    r = await client.post(
        f"{ROBOFLOW_URL}/{ROBOFLOW_MODEL}",
        params={
            "api_key": ROBOFLOW_API_KEY,
            "confidence": round(CONFIDENCE * 100),
            "overlap": ROBOFLOW_OVERLAP,
            "format": "json",
        },
        content=base64.b64encode(jpeg),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if r.status_code == 401:
        raise RuntimeError("Roboflow rejected the API key (401)")
    if r.status_code == 404:
        raise RuntimeError(
            f"Roboflow model {ROBOFLOW_MODEL!r} not found (404) - "
            "it must be 'model-slug/version', e.g. 'coco/9'"
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


def plausible(track: Track) -> bool:
    """Is this track's velocity physically possible for what it claims to be?"""
    speed = float(np.linalg.norm(track.vel))
    return speed <= IMPLAUSIBLE_SPEED_MPS.get(track.kind, 32.0)


def assess(
    tracks: list[Track], seen_before: set | None = None,
) -> tuple[str, list[dict], set]:
    """Return (risk, conflicts, pairs_in_conflict_now).

    `seen_before` is the pair set from the previous assessment. A conflict is
    only reported once it has persisted across consecutive assessments, so a
    single frame of bad association cannot raise an alarm on its own.
    """
    usable = [t for t in tracks if t.hits >= MIN_HITS_FOR_CONFLICT and plausible(t)]
    vehicles = [t for t in usable if t.kind == "vehicle"]
    peds = [t for t in usable if t.kind == "pedestrian"]

    candidates, pairs = [], set()
    for v in vehicles:
        for p in peds:
            t, miss = closest_approach(v.pos, v.vel, p.pos, p.vel)
            if 0 < t <= TTC_HORIZON_S and miss <= MISS_DISTANCE_M:
                pair = (v.track_id, p.track_id)
                pairs.add(pair)
                candidates.append({
                    "vehicle_track": v.track_id,
                    "pedestrian_track": p.track_id,
                    "seconds_to_closest_approach": round(t, 2),
                    "miss_distance_m": round(miss, 2),
                })

    prior = seen_before or set()
    confirmed = [
        c for c in candidates
        # Imminent conflicts pass straight through; marginal ones must repeat.
        if c["seconds_to_closest_approach"] <= IMMINENT_TTC_S
        or CONFLICT_PERSISTENCE <= 1
        or (c["vehicle_track"], c["pedestrian_track"]) in prior
    ]

    confirmed.sort(key=lambda c: c["seconds_to_closest_approach"])
    if not confirmed:
        return "CLEAR", [], pairs
    soonest = confirmed[0]["seconds_to_closest_approach"]
    level = "HIGH" if soonest <= 3.0 else "MEDIUM" if soonest <= 5.0 else "LOW"
    return level, confirmed[:5], pairs


async def loop() -> None:
    """Poll → detect → track → assess. Every failure path degrades, never raises."""
    h_mat = _homography(_parse_quad(SRC_QUAD), _parse_quad(DST_QUAD))
    cam: dict = {}
    last_frame_hash: bytes | None = None
    prior_pairs: set = set()

    async with httpx.AsyncClient(follow_redirects=True) as client:
        while True:
            try:
                if time.time() < REPLAY_UNTIL:
                    await asyncio.sleep(POLL_SECONDS)
                    continue
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

                # Refetching the same frame is not harmless. Identical
                # detections mean a measured velocity of exactly zero, and the
                # smoothing would drag every track toward stationary -
                # systematically under-estimating closing speed and
                # suppressing real conflicts. Skip before spending inference.
                digest = hashlib.sha256(jpeg).digest()
                if digest == last_frame_hash:
                    STATE.duplicate_frames += 1
                    await asyncio.sleep(POLL_SECONDS)
                    continue
                # NB: the hash is committed only after the cycle succeeds, at
                # the bottom of this block. Committing it here meant a failed
                # inference on a static frame was never retried - the next poll
                # saw a duplicate, skipped, and the system sat in a stale state
                # instead of escalating to FAIL_SAFE. Found by chaos_test.py.

                if not ROBOFLOW_API_KEY:
                    STATE.status = "FAIL_SAFE"
                    STATE.reason = "ROBOFLOW_API_KEY not set - detection disabled"
                    STATE.risk = "UNKNOWN"
                    STATE.frames += 1
                    STATE.updated = time.time()
                    await asyncio.sleep(POLL_SECONDS)
                    continue

                frame = upscale_jpeg(jpeg, UPSCALE)
                iw, ih = jpeg_dimensions(frame)
                preds = await detect(client, frame)
                dets = ground_positions(preds, float(iw), float(ih), h_mat)

                now = time.time()
                tracks = TRACKER.update(dets, now)
                risk, conflicts, prior_pairs = assess(tracks, prior_pairs)

                STATE.status = "ACTIVE"
                STATE.reason = "ok"
                STATE.risk = risk
                STATE.conflicts = conflicts
                if risk in ("HIGH", "MEDIUM") and conflicts:
                    STATE.near_misses.append({
                        "at": now,
                        "risk": risk,
                        "ttc": conflicts[0]["seconds_to_closest_approach"],
                        "miss_m": conflicts[0]["miss_distance_m"],
                    })
                    # Bounded; this is a live signal, not an archive.
                    del STATE.near_misses[:-200]
                    record_conflict(
                        STATE.camera.get("id", "unknown"),
                        STATE.camera.get("name", "unknown"), risk,
                        conflicts[0]["seconds_to_closest_approach"],
                        conflicts[0]["miss_distance_m"], now)
                STATE.counts = {
                    "vehicles": sum(1 for t in tracks if t.kind == "vehicle"),
                    "pedestrians": sum(1 for t in tracks if t.kind == "pedestrian"),
                    "detections": len(dets),
                }
                # Calibration self-check. Human walking speed is ~1.4 m/s, so
                # the median observed pedestrian speed divided by 1.4 is the
                # scale error in the ground-plane calibration - a way to
                # measure the homography against a known physical constant
                # rather than trusting an eyeballed quad.
                # Use the 90th percentile, not the median. At a signalised
                # crossing most pedestrians are standing still waiting for the
                # light, and they drag a median toward zero for entirely real
                # reasons. The fastest movers are the ones actually crossing,
                # and those are the ones whose speed should equal 1.4 m/s if
                # the metric scale is right.
                ped_speeds = sorted(
                    float(np.linalg.norm(t.vel)) for t in tracks
                    if t.kind == "pedestrian" and t.hits >= 2)
                if ped_speeds:
                    idx = min(len(ped_speeds) - 1,
                              int(round(0.9 * (len(ped_speeds) - 1))))
                    STATE.median_ped_speed = round(ped_speeds[idx], 2)
                STATE.frames += 1
                STATE.last_ok = now
                STATE.consecutive_errors = 0
                STATE.updated = now
                # Only now is this frame truly consumed.
                last_frame_hash = digest
                await emit_alert(client)

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
                await emit_alert(client)

            await asyncio.sleep(POLL_SECONDS)



# --- alerting ---------------------------------------------------------------

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_AUTH_HEADER = os.getenv("WEBHOOK_AUTH_HEADER", "Authorization")
WEBHOOK_AUTH_VALUE = os.getenv("WEBHOOK_AUTH_VALUE", "")
# Floor between posts, so a risk level oscillating on the boundary cannot
# machine-gun the events dashboard and bury the alerts that matter.
WEBHOOK_MIN_INTERVAL_S = float(os.getenv("WEBHOOK_MIN_INTERVAL_S", "3.0"))

# Intervention is chosen by how much time is actually left for someone to act,
# not by risk label. The two are not the same, and conflating them gets the
# design backwards.
#
# At a 25mph limit a driver needs ~2.7s to perceive, react, and stop. Our own
# detection latency is ~2.9s (2.0s camera refresh + inference + actuation). So
# a warning aimed at a human only helps if the conflict is spotted more than
# ~5.6s out. Below that, no human can use it.
#
#   >= 5.6s   human-actionable   direct attention to the crosswalk
#   2.9-5.6s  machine-only       hold the signal; needs nobody to react
#   < 2.9s    do not alert       see below
#
# That last band matters. Warning someone with less time than they need is not
# merely useless - a startled pedestrian mid-crossing may freeze rather than
# clear, and a driver who flinches at a light may swerve. Below the reaction
# floor the system records the event for analysis and stays silent.
HUMAN_REACTION_S = 2.7
DETECTION_LATENCY_S = 2.9
HUMAN_ACTIONABLE_TTC_S = HUMAN_REACTION_S + DETECTION_LATENCY_S   # 5.6


def choose_intervention(risk: str, ttc: float | None) -> tuple[str, str]:
    """Return (action, why) from the time actually remaining."""
    if risk in ("CLEAR", "UNKNOWN") or ttc is None:
        # UNKNOWN never requests an intervention. When the system cannot see
        # the street the grid falls back to normal fixed timing.
        return "No_Action", "no conflict, or nothing reliable to say"

    remaining = ttc - DETECTION_LATENCY_S
    if ttc >= HUMAN_ACTIONABLE_TTC_S:
        return ("Activate_LED_Crosswalk",
                f"{remaining:.1f}s left after latency, above the {HUMAN_REACTION_S}s "
                "a driver needs - in-pavement LEDs draw the eye to the crosswalk "
                "itself, adding no message to read")
    if remaining > 0:
        return ("Extend_All_Red_5s",
                f"only {remaining:.1f}s left, below human reaction time - hold "
                "the signal, which removes the conflict without anyone reacting")
    return ("Log_Only",
            "already inside the reaction floor; a warning now could startle "
            "rather than help, so record it and stay silent")
SEVERITY = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low",
            "CLEAR": "low", "UNKNOWN": "low"}
# UNKNOWN sits at the bottom so losing the feed never counts as an
# escalation and never fast-paths an alert past the throttle.
RISK_ORDER = {"UNKNOWN": 0, "CLEAR": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def build_alert() -> dict:
    """The webhook payload. Mirrors docs/roboflow-agent-prompt.md."""
    top = STATE.conflicts[0] if STATE.conflicts else None
    ttc = top["seconds_to_closest_approach"] if top else None
    action, why = choose_intervention(STATE.risk, ttc)
    return {
        "event_type": "collision_warning",
        "schema_version": "ghost-v2x.v1",
        "camera_id": STATE.camera.get("id"),
        "camera_name": STATE.camera.get("name"),
        "status": STATE.status,
        "reason": STATE.reason,
        "risk_level": STATE.risk,
        "severity": SEVERITY.get(STATE.risk, "low"),
        "ttc_seconds": top["seconds_to_closest_approach"] if top else None,
        "miss_distance_m": top["miss_distance_m"] if top else None,
        "recommended_action": action,
        "action_rationale": why,
        "collision_warning": STATE.risk in ("HIGH", "MEDIUM"),
        "vehicle_count": STATE.counts.get("vehicles", 0),
        "pedestrian_count": STATE.counts.get("pedestrians", 0),
        "alert_description": (
            f"Ghost-V2X {STATE.risk}: {action}"
            + (f"; TTC={top['seconds_to_closest_approach']}s"
               f"; miss={top['miss_distance_m']}m" if top else "")
            + f"; vehicles={STATE.counts.get('vehicles', 0)}"
            f"; pedestrians={STATE.counts.get('pedestrians', 0)}"
            + ("" if STATE.status == "ACTIVE" else f"; {STATE.status}: {STATE.reason}")
        ),
    }


async def emit_alert(client: httpx.AsyncClient, force: bool = False) -> None:
    """Post on transitions only. Never raises - alerting must not stop sensing."""
    global _last_alert_key, _last_alert_at
    if not WEBHOOK_URL:
        return
    key = (STATE.status, STATE.risk)
    now = time.time()
    if not force:
        if key == _last_alert_key:
            return
        # An anti-flap throttle must never delay an escalation. Risk climbing
        # toward HIGH is the one event that has to go out immediately; only
        # de-escalation and same-level churn are worth rate limiting.
        previous = _last_alert_key[1] if _last_alert_key else "CLEAR"
        escalating = RISK_ORDER.get(STATE.risk, 0) > RISK_ORDER.get(previous, 0)
        if not escalating and now - _last_alert_at < WEBHOOK_MIN_INTERVAL_S:
            return
    _last_alert_key, _last_alert_at = key, now

    headers = {"Content-Type": "application/json"}
    if WEBHOOK_AUTH_VALUE:
        headers[WEBHOOK_AUTH_HEADER] = WEBHOOK_AUTH_VALUE
    try:
        r = await client.post(WEBHOOK_URL, json=build_alert(),
                              headers=headers, timeout=10)
        STATE.alerts_sent += 1
        if r.status_code >= 400:
            STATE.alert_error = f"HTTP {r.status_code}: {r.text[:120]}"
            log.warning("alert POST rejected: %s", STATE.alert_error)
        else:
            STATE.alert_error = ""
            log.info("alert sent: %s / %s", STATE.status, STATE.risk)
    except Exception as exc:
        STATE.alert_error = f"{type(exc).__name__}: {exc}"
        log.warning("alert POST failed: %s", STATE.alert_error)


_last_alert_key: tuple | None = None
_last_alert_at: float = 0.0


# --- replay -----------------------------------------------------------------

REPLAY_UNTIL: float = 0.0


async def replay_scenario() -> None:
    """Drive a synthetic near-miss through the real pipeline.

    A live demo should not depend on Harlem producing a genuine near-miss
    during the ninety seconds you are on stage. This injects a car closing on
    a crossing pedestrian and runs it through the same Tracker, the same
    closest-approach physics, and the same alerting path as live traffic -
    only the detections are synthetic.
    """
    global REPLAY_UNTIL
    log.info("replay: synthetic near-miss starting")
    tracker = Tracker()
    replay_pairs: set = set()
    start = time.time()
    REPLAY_UNTIL = start + 30.0

    async with httpx.AsyncClient() as client:
        for step in range(10):
            t = step * 1.6
            # Car north at 8 m/s; pedestrian east at 1.4 m/s, timed to meet.
            car = np.array([10.0, 40.0 - 8.0 * t])
            ped = np.array([4.4 + 1.4 * t, 8.0])
            now = start + t
            tracks = tracker.update(
                [("vehicle", car), ("pedestrian", ped)], now)
            risk, conflicts, replay_pairs = assess(tracks, replay_pairs)

            STATE.status = "ACTIVE"
            STATE.reason = "REPLAY - synthetic scenario, not live traffic"
            STATE.risk = risk
            STATE.conflicts = conflicts
            STATE.counts = {"vehicles": 1, "pedestrians": 1, "detections": 2}
            STATE.frames += 1
            STATE.last_ok = time.time()
            STATE.updated = time.time()
            REPLAY_UNTIL = time.time() + 4.0

            await emit_alert(client)
            await asyncio.sleep(1.2)

    REPLAY_UNTIL = 0.0
    TRACKER.tracks.clear()
    log.info("replay: finished, returning to live traffic")


# --- web -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(title="Ghost-V2X", lifespan=lifespan)

_static = Path(__file__).parent / "static"
if _static.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


# Google's frontend reserves /healthz and 404s it before it ever reaches the
# container - verified against both Cloud Run host forms while the same route
# returned 200 locally. /health is the reachable one; /healthz stays registered
# so local tooling and the chaos suite keep working.
@app.get("/health")
@app.get("/healthz")
def healthz():
    """Liveness only. The pipeline degrades to FAIL_SAFE without the container
    being unhealthy - Cloud Run must not restart us for a dark camera."""
    return {"ok": True}


@app.post("/api/replay")
async def api_replay():
    """Trigger the synthetic near-miss. Safe to hit live during a demo."""
    if time.time() < REPLAY_UNTIL:
        return JSONResponse({"ok": False, "error": "replay already running"}, 409)
    asyncio.create_task(replay_scenario())
    return {"ok": True, "message": "synthetic near-miss running for ~16s"}


@app.get("/api/alert")
def api_alert():
    """Exactly what would be POSTed right now - handy for testing the webhook."""
    return JSONResponse(build_alert())


@app.get("/insights", response_class=HTMLResponse)
def insights_page():
    """Who this is for: EMS in seconds, DOT in years, counsel ongoing."""
    path = Path(__file__).parent / "insights.html"
    if not path.exists():
        return HTMLResponse("<h1>insights.html missing</h1>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/api/insights")
def api_insights():
    """Per-intersection interventions, derived from NYPD contributing factors.

    Built offline by build_insights.py so the page never waits on Socrata.
    """
    path = Path(__file__).parent / "insights.json"
    if not path.exists():
        return JSONResponse({"intersections": []})
    return JSONResponse(
        {"intersections": json.loads(path.read_text(encoding="utf-8"))})


@app.get("/map", response_class=HTMLResponse)
def map_page():
    """Risk map: crash history, live status, and the near-miss signal."""
    path = Path(__file__).parent / "map.html"
    if not path.exists():
        return HTMLResponse("<h1>map.html missing</h1>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/api/map-data")
def api_map_data():
    """Everything the map needs: history, live status, and the leading signal.

    Three layers, deliberately distinct:
      - `injured`  historical harm from NYC's crash record. Lagging.
      - `live`     what the sensor sees right now at the monitored camera.
      - `near_misses` conflicts observed without a crash. Leading.

    An intersection with few recorded crashes but a rising near-miss count is
    where the next one happens, and that is the layer only live detection can
    produce.
    """
    ranked = []
    path = Path(__file__).parent / "camera_risk_ranking.json"
    if path.exists():
        try:
            ranked = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("could not read ranking: %s", exc)

    now = time.time()
    recent = [n for n in STATE.near_misses if now - n["at"] <= 3600]
    observed = conflict_summary()
    return JSONResponse({
        "observed": observed,
        "monitored_camera_id": STATE.camera.get("id"),
        "live": {
            "status": STATE.status,
            "risk": STATE.risk,
            "camera": STATE.camera.get("name"),
            "vehicles": STATE.counts.get("vehicles", 0),
            "pedestrians": STATE.counts.get("pedestrians", 0),
            "conflicts": STATE.conflicts,
            "near_miss_last_hour": len(recent),
            "near_miss_total": len(STATE.near_misses),
            "scale_error": STATE.scale_error_value(),
        },
        "near_misses": recent[-40:],
        "cameras": [
            {
                "id": c["id"], "name": c["name"], "area": c.get("area"),
                "lat": float(c["latitude"]), "lon": float(c["longitude"]),
                "injured": c["people_injured"], "crashes": c["crashes"],
            }
            for c in ranked
            if c.get("latitude") and c.get("longitude")
        ],
    })


@app.get("/api/state")
def api_state():
    return JSONResponse(STATE.as_dict())


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Live operator view.

    The alert rail takes zero vertical space until risk is MEDIUM or HIGH, so
    its arrival is itself the signal - the operator does not have to read
    anything to know something changed.
    """
    path = Path(__file__).parent / "dashboard.html"
    if not path.exists():
        return HTMLResponse("<h1>dashboard.html missing</h1>", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))
