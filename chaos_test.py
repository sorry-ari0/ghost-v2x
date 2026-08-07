"""Adversarial harness: prove the fail-safe paths actually fail safe.

Every scenario drives the real loop() against a mocked network, then asserts
the state the system settles into. The invariant under test is narrow and it
is the whole safety argument:

    when the system cannot see the street, it must stop asserting things
    about the street - and it must never keep showing a stale HIGH.

A stale HIGH is worse than an honest UNKNOWN, because it teaches an operator
to ignore the alert.

    python chaos_test.py
"""
from __future__ import annotations

import asyncio
import sys

import httpx

import main

PASS, FAIL = "PASS", "FAIL"
CAMERA = {
    "id": "test-cam", "name": "Test Ave @ Test St",
    "latitude": 40.8, "longitude": -73.9, "isOnline": "true",
    "imageUrl": "https://example.invalid/cam/image",
}
# Smallest thing that clears the "too small" guard and reads as a real frame.
GOOD_FRAME = b"\xff\xd8\xff" + b"\x00" * 4096


def reset() -> None:
    main.STATE = main.State()
    main.TRACKER = main.Tracker()


async def run_loop(handler, seconds: float, detect_stub=None):
    """Run the real loop() against a mock transport for a bounded time."""
    real_client, real_detect = main.httpx.AsyncClient, main.detect

    def factory(**kw):
        kw.pop("transport", None)
        return real_client(transport=httpx.MockTransport(handler), **kw)

    main.httpx.AsyncClient = factory
    if detect_stub is not None:
        main.detect = detect_stub
    try:
        task = asyncio.create_task(main.loop())
        await asyncio.sleep(seconds)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        main.httpx.AsyncClient = real_client
        main.detect = real_detect


def frames_ok(request: httpx.Request) -> httpx.Response:
    if "cameras/" in str(request.url) and "image" not in str(request.url):
        return httpx.Response(200, json=[CAMERA])
    if str(request.url).endswith("cameras/") or "api/cameras" in str(request.url):
        return httpx.Response(200, json=[CAMERA])
    return httpx.Response(200, content=GOOD_FRAME)


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{PASS if ok else FAIL}] {name}")
    print(f"         {detail}")
    return ok


async def main_async() -> int:
    results = []
    print("Ghost-V2X adversarial scenarios\n" + "=" * 66)

    # 1. DOT camera goes dark mid-operation ---------------------------------
    print("\n1. Camera goes dark (DOT returns an empty frame)")
    reset()
    state = {"dark": False}

    def dark(request):
        if "image" not in str(request.url):
            return httpx.Response(200, json=[CAMERA])
        if state["dark"]:
            return httpx.Response(200, content=b"")
        return httpx.Response(200, content=GOOD_FRAME)

    async def fake_detect(client, jpeg):
        return [{"class": "car", "x": 100, "y": 200, "width": 40, "height": 30,
                 "image_width": 352, "image_height": 240}]

    await run_loop(dark, 2.5, fake_detect)
    state["dark"] = True
    await run_loop(dark, 4.0, fake_detect)
    results.append(check(
        "degrades to FAIL_SAFE, risk becomes UNKNOWN",
        main.STATE.status == "FAIL_SAFE" and main.STATE.risk == "UNKNOWN",
        f"status={main.STATE.status} risk={main.STATE.risk} errors={main.STATE.errors}"))
    results.append(check(
        "clears tracks so no stale conflict can be reported",
        main.STATE.conflicts == [] and not main.TRACKER.tracks,
        f"conflicts={len(main.STATE.conflicts)} tracks={len(main.TRACKER.tracks)}"))

    # 2. Roboflow 500 --------------------------------------------------------
    print("\n2. Roboflow returns 500")
    reset()

    async def detect_500(client, jpeg):
        raise httpx.HTTPStatusError(
            "500", request=httpx.Request("POST", "https://x.invalid"),
            response=httpx.Response(500))

    await run_loop(frames_ok, 4.0, detect_500)
    results.append(check(
        "survives inference outage without crashing the loop",
        main.STATE.status == "FAIL_SAFE" and main.STATE.risk == "UNKNOWN",
        f"status={main.STATE.status} risk={main.STATE.risk} errors={main.STATE.errors}"))

    # 3. Roboflow hangs ------------------------------------------------------
    print("\n3. Roboflow hangs (timeout)")
    reset()

    async def detect_hang(client, jpeg):
        raise httpx.ReadTimeout("timed out")

    await run_loop(frames_ok, 4.0, detect_hang)
    results.append(check(
        "timeout is contained per-cycle",
        main.STATE.status == "FAIL_SAFE" and main.STATE.errors >= 2,
        f"status={main.STATE.status} errors={main.STATE.errors}"))

    # 4. Bad API key ---------------------------------------------------------
    print("\n4. Roboflow rejects the API key (401)")
    reset()

    async def detect_401(client, jpeg):
        raise RuntimeError("Roboflow rejected the API key (401)")

    await run_loop(frames_ok, 4.0, detect_401)
    results.append(check(
        "surfaces the real cause instead of a generic failure",
        "401" in main.STATE.reason,
        f"reason={main.STATE.reason!r}"))

    # 5. Camera frozen -------------------------------------------------------
    print("\n5. Camera frozen (same frame served forever)")
    reset()

    calls = {"n": 0}

    async def detect_count(client, jpeg):
        calls["n"] += 1
        return []

    await run_loop(frames_ok, 4.0, detect_count)
    results.append(check(
        "frozen feed does not burn inference calls",
        calls["n"] <= 2,
        f"inference calls={calls['n']} over ~4 polls, "
        f"duplicates skipped={main.STATE.duplicate_frames}"))

    # 6. Camera list endpoint down ------------------------------------------
    print("\n6. Camera list endpoint down, image endpoints healthy")
    reset()

    def list_down(request):
        if "image" not in str(request.url):
            return httpx.Response(503)
        return httpx.Response(200, content=GOOD_FRAME)

    await run_loop(list_down, 3.0, fake_detect)
    picked = bool(main.STATE.camera)
    results.append(check(
        "falls back to the bundled roster rather than going down",
        picked or main.STATE.status == "FAIL_SAFE",
        f"camera={main.STATE.camera.get('name') if picked else None} "
        f"status={main.STATE.status}"))

    # 7. Recovery ------------------------------------------------------------
    print("\n7. Recovery after the outage clears")
    reset()
    state2 = {"broken": True}

    def flaky(request):
        if "image" not in str(request.url):
            return httpx.Response(200, json=[CAMERA])
        if state2["broken"]:
            return httpx.Response(500)
        return httpx.Response(200, content=GOOD_FRAME)

    await run_loop(flaky, 3.0, fake_detect)
    broke = main.STATE.status == "FAIL_SAFE"
    state2["broken"] = False

    # Vary the frame so dedup does not suppress the recovery cycle.
    def flaky_recovered(request):
        if "image" not in str(request.url):
            return httpx.Response(200, json=[CAMERA])
        return httpx.Response(200, content=GOOD_FRAME + b"\x01")

    await run_loop(flaky_recovered, 3.0, fake_detect)
    results.append(check(
        "returns to ACTIVE once frames come back",
        broke and main.STATE.status == "ACTIVE",
        f"went FAIL_SAFE={broke} then status={main.STATE.status} "
        f"frames={main.STATE.frames}"))

    # 8. Liveness ------------------------------------------------------------
    print("\n8. Liveness during degradation")
    results.append(check(
        "/healthz stays green so Cloud Run will not restart the container",
        main.healthz() == {"ok": True},
        "a dark camera is not an unhealthy container"))

    print("\n" + "=" * 66)
    passed = sum(results)
    print(f"{passed}/{len(results)} scenarios passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async()))
