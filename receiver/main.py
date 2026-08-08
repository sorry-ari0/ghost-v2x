"""Mock traffic signal controller for Ghost-V2X.

Stands in for the thing that would actually act on an alert: a signal
controller, a crosswalk LED driver, or an operations desk. It exists to show
the loop closing on real infrastructure rather than ending at a log line.

Deliberately dumb. It does not compute risk, does not second-guess the
recommended action, and does not filter. Ghost-V2X owns that reasoning; a
downstream actuator that re-litigates it is how two systems end up disagreeing
about whether to hold a signal.
"""
from __future__ import annotations

import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="Ghost-V2X Signal Controller")

# In-memory and bounded. This is a demonstration actuator, not a record of
# account; the durable conflict log lives in the sensor.
ALERTS: deque = deque(maxlen=25)
STARTED = time.time()


@app.post("/webhook")
@app.post("/webhook/")
async def receive(request: Request):
    """Accept an alert and report the action taken."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    action = payload.get("recommended_action", "No_Action")
    alert = {
        "at": time.time(),
        "camera_id": payload.get("camera_id", "unknown"),
        "camera_name": payload.get("camera_name") or payload.get("camera_id", "unknown"),
        "status": payload.get("status", "UNKNOWN"),
        "risk_level": payload.get("risk_level", "UNKNOWN"),
        "ttc_seconds": payload.get("ttc_seconds"),
        "miss_distance_m": payload.get("miss_distance_m"),
        "recommended_action": action,
        "rationale": payload.get("action_rationale", ""),
        "vehicles": payload.get("vehicle_count", 0),
        "pedestrians": payload.get("pedestrian_count", 0),
    }
    ALERTS.appendleft(alert)

    # flush=True or Cloud Run buffers stdout and the line never appears live.
    print(f"[ALERT] {alert['risk_level']:<7} ttc={alert['ttc_seconds']} "
          f"-> {action}", flush=True)
    return {"status": "acknowledged", "action_taken": action}


@app.get("/api/alerts")
def api_alerts():
    return JSONResponse({
        "system_status": "ONLINE",
        "active_intersections": len({a["camera_id"] for a in ALERTS}) or 1,
        "uptime_s": round(time.time() - STARTED),
        "recent_alerts": list(ALERTS),
    })


@app.get("/health")
@app.get("/healthz")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return PAGE


PAGE = """<style>
  :root{color-scheme:dark}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;
       background:#0b0e14;color:#c8d0e0;padding:2.5rem 2rem 4rem;
       max-width:940px;margin:0 auto}
  h1{font-size:1rem;letter-spacing:.16em;text-transform:uppercase;
     color:#7d8799;margin:0 0 .25rem}
  .sub{color:#5a6373;font-size:.82rem;margin-bottom:2rem;max-width:64ch}
  .status{display:flex;gap:2.5rem;flex-wrap:wrap;margin-bottom:2.5rem}
  .stat .n{font-size:1.9rem;font-weight:700;letter-spacing:-.02em}
  .stat .l{color:#5a6373;font-size:.7rem;text-transform:uppercase;
           letter-spacing:.1em}
  .on{color:#3ddc97}
  .feed{border:1px solid #1f2634;border-radius:10px;overflow:hidden}
  .row{display:grid;grid-template-columns:88px 1fr auto;gap:1rem;
       padding:.85rem 1.1rem;border-bottom:1px solid #141a24;align-items:center}
  .row:last-child{border-bottom:0}
  .row.HIGH{background:linear-gradient(90deg,#1e0c10,transparent 60%)}
  .row.MEDIUM{background:linear-gradient(90deg,#1e150c,transparent 60%)}
  .lvl{font-weight:700;font-size:.85rem;letter-spacing:.06em}
  .CLEAR .lvl{color:#3ddc97}.LOW .lvl{color:#ffd166}
  .MEDIUM .lvl{color:#ff9f45}.HIGH .lvl{color:#ff4d5e}
  .UNKNOWN .lvl{color:#5a6373}
  .act{font-size:.95rem;color:#e8edf5}
  .why{color:#6b7688;font-size:.74rem;margin-top:.2rem;line-height:1.45}
  .meta{color:#5a6373;font-size:.72rem;text-align:right;white-space:nowrap}
  .empty{padding:2.5rem 1.1rem;color:#5a6373;text-align:center}
  .note{color:#5a6373;font-size:.76rem;margin-top:2rem;max-width:70ch;
        line-height:1.6}
  a{color:#4de2ff}
</style>

<h1>Signal Controller</h1>
<div class="sub">
  Downstream actuator for Ghost-V2X. Receives conflict alerts and reports the
  action taken. It does not compute risk &mdash; the sensor owns that, and an
  actuator that second-guesses it is how two systems end up disagreeing about
  whether to hold a signal.
</div>

<div class="status">
  <div class="stat"><div class="n on" id="sys">&mdash;</div>
    <div class="l">System</div></div>
  <div class="stat"><div class="n" id="cnt">&mdash;</div>
    <div class="l">Alerts received</div></div>
  <div class="stat"><div class="n" id="ints">&mdash;</div>
    <div class="l">Intersections</div></div>
  <div class="stat"><div class="n" id="up">&mdash;</div>
    <div class="l">Uptime</div></div>
</div>

<div class="feed" id="feed">
  <div class="empty">waiting for alerts&hellip;</div>
</div>

<div class="note">
  Actions are chosen by the sensor from the time actually remaining, not from
  the risk label. Above ~5.6s a person can still react, so the crosswalk lights.
  Between ~2.9s and 5.6s only the signal can act. Below that the system stays
  silent on purpose &mdash; a warning with less time than a human needs can
  startle rather than help.
  <br><br>
  <a href="https://ghost-v2x-73791867861.us-east1.run.app/">Live sensor</a> &middot;
  <a href="https://ghost-v2x-73791867861.us-east1.run.app/map">Planning map</a> &middot;
  <a href="/api/alerts">Raw JSON</a>
</div>

<script>
const esc = s => String(s ?? '').replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ago = t => { const d = Math.round(Date.now()/1000 - t);
  return d < 60 ? d + 's ago' : Math.round(d/60) + 'm ago'; };

async function tick(){
  let d;
  try { d = await (await fetch('/api/alerts')).json(); } catch(e){ return; }

  document.getElementById('sys').textContent  = d.system_status;
  document.getElementById('cnt').textContent  = d.recent_alerts.length;
  document.getElementById('ints').textContent = d.active_intersections;
  document.getElementById('up').textContent   =
    d.uptime_s < 3600 ? Math.round(d.uptime_s/60) + 'm'
                      : Math.round(d.uptime_s/3600) + 'h';

  const feed = document.getElementById('feed');
  if (!d.recent_alerts.length){
    feed.innerHTML = '<div class="empty">waiting for alerts&hellip;</div>';
    return;
  }
  feed.innerHTML = d.recent_alerts.map(a => `
    <div class="row ${esc(a.risk_level)}">
      <div class="lvl">${esc(a.risk_level)}</div>
      <div>
        <div class="act">${esc(a.recommended_action)}</div>
        ${a.rationale ? `<div class="why">${esc(a.rationale)}</div>` : ''}
      </div>
      <div class="meta">
        ${a.ttc_seconds != null ? 'TTC ' + esc(a.ttc_seconds) + 's<br>' : ''}
        ${a.miss_distance_m != null ? 'miss ' + esc(a.miss_distance_m) + 'm<br>' : ''}
        ${esc(ago(a.at))}
      </div>
    </div>`).join('');
}
tick(); setInterval(tick, 1500);
</script>
"""
