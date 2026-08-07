# Prompt for the Roboflow agent

Paste everything below the line.

---

Context before you change anything.

Ghost-V2X is a collision-risk service running on Google Cloud Run. It polls a
live NYC DOT traffic camera (Lenox Ave @ 125 St, id
`156b0613-239a-4e77-aa0e-0a4becfc0b05`), sends each frame to Roboflow hosted
inference, and decides risk itself. Roboflow Vision Events is a **sink** for
alerts and annotated frames, not the place risk is computed.

**Division of labor, and please do not blur it:**

- **Roboflow does detection.** Bounding boxes and classes for one frame.
- **Cloud Run does risk.** It projects each box's bottom-centre through a
  homography onto a metric ground plane, tracks objects across frames in
  metres, and computes closest point of approach from relative velocity.

This matters because a single frame cannot produce a time-to-collision.
TTC needs velocity, velocity needs at least two frames, and proximity measured
in **image space is not proximity in the real world** — a pedestrian on the
near sidewalk and a car across the intersection overlap in pixels constantly
while being fifty feet apart. That is the central correctness problem in this
system and the homography is how it is solved.

So: the webhook payload is **produced by Cloud Run and posted to you**. Nothing
in the Roboflow workflow should compute `risk_level`, `ttc_seconds`, or
`recommended_action`.

## Three changes needed

**1. `risk_level` must include `UNKNOWN`, and `NO_CONFLICT` should be `CLEAR`.**

The current enum is `NO_CONFLICT | LOW | MEDIUM | HIGH`. The service actually
emits:

```
CLEAR | LOW | MEDIUM | HIGH | UNKNOWN
```

`UNKNOWN` is not a rounding detail, it is the safety design. Whenever the
service loses confidence — dark camera, inference outage, bad key, frozen feed
— it stops asserting anything about the street and reports `UNKNOWN`. A stale
`HIGH` is dangerous because it trains operators to ignore alerts; a silent
`NO_CONFLICT` during an outage is worse, because it actively reassures them
that an unwatched intersection is safe.

With the present enum there is no way to distinguish "nobody is in danger"
from "I cannot see." Please add `UNKNOWN` and rename `NO_CONFLICT` to `CLEAR`.

**2. Drop `basis=current_frame_proximity` from `alert_description`.**

That string says the risk came from proximity within one frame. It did not,
and it must not. Let Cloud Run supply the whole `alert_description` string
rather than composing it in the workflow.

**3. Add a `status` field alongside `risk_level`.**

```
"status": "ACTIVE | FAIL_SAFE | STARTING"
```

So the dashboard can show *why* risk is `UNKNOWN`. Pair it with the existing
free-text reason:

```
"reason": "camera returned an empty frame"
```

## Target payload

```json
{
  "event_type": "collision_warning",
  "schema_version": "ghost-v2x.v1",
  "camera_id": "156b0613-239a-4e77-aa0e-0a4becfc0b05",
  "camera_name": "Lenox Ave @ 125 St",
  "status": "ACTIVE",
  "reason": "ok",
  "risk_level": "HIGH",
  "severity": "high",
  "ttc_seconds": 3.42,
  "miss_distance_m": 1.4,
  "recommended_action": "Extend_All_Red_5s",
  "collision_warning": true,
  "vehicle_count": 2,
  "pedestrian_count": 1,
  "alert_description": "Ghost-V2X HIGH: Extend_All_Red_5s; TTC=3.42s; miss=1.4m; vehicles=2; pedestrians=1"
}
```

`miss_distance_m` is new and worth surfacing: closest-point-of-approach
distance is what separates a genuine near-miss from two objects that merely
pass near each other. A 0.5s TTC with an 8m miss is not a conflict.

## What we need back from you

The blocker is that 0 events have landed, and we cannot feed you until we know
exactly where to POST. Please return:

1. **The exact webhook endpoint URL** to POST to.
2. **The exact auth header** — header name and whether it takes the workspace
   API key directly or a separate Vision Events token.
3. **Whether our key has `vision-events:write`.** If you cannot read the scope,
   say so plainly and tell us how to mint a key that has it — do not guess.
4. **Confirmation that `UNKNOWN` and `CLEAR` are accepted** by the updated
   schema, and what you do with an event whose `collision_warning` is `false`
   (stored, or dropped?). We will be sending `UNKNOWN` events during
   degradation and they must not be silently discarded — those are the ones
   that prove the fail-safe works.
5. **A single working `curl`** that posts one sample event successfully, so we
   can verify the path end-to-end before wiring it into the service.

If any of this is not knowable from where you sit, say which parts and why,
rather than producing a plausible-looking endpoint we will waste time on.
