"""Find a Roboflow model that actually works on our frames.

The Roboflow agent could not run this - it has no shell and no raw HTTP client.
You do. This settles three open questions in one pass:

  1. which model slug resolves under your key on detect.roboflow.com
  2. whether it returns BOTH a person class and a vehicle class
  3. what image dimensions come back, so we can confirm the coordinate frame

Run it in Cloud Shell:

    export ROBOFLOW_API_KEY=<your key>
    python3 probe_model.py

It prints the exact line to paste into your gcloud deploy.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request

CAMERA = "156b0613-239a-4e77-aa0e-0a4becfc0b05"
FRAME_URL = f"https://webcams.nyctmc.org/api/cameras/{CAMERA}/image"

# Ordered by how likely they are to work, per Roboflow's own docs.
CANDIDATES = ["coco/38", "rfdetr-nano", "coco/9", "coco/1"]
HOSTS = ["https://detect.roboflow.com", "https://serverless.roboflow.com"]

VEHICLES = {"car", "truck", "bus", "motorbike", "motorcycle", "van"}
PEOPLE = {"person", "pedestrian", "bicycle", "cyclist"}


def get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ghost-v2x"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def upscale(jpeg: bytes, factor: float = 2.0) -> bytes:
    from PIL import Image
    img = Image.open(io.BytesIO(jpeg))
    big = img.resize((round(img.width * factor), round(img.height * factor)),
                     Image.LANCZOS)
    buf = io.BytesIO()
    big.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def try_model(host: str, model: str, frame: bytes, key: str):
    url = f"{host}/{model}?" + urllib.parse.urlencode(
        {"api_key": key, "confidence": 25, "overlap": 45})
    req = urllib.request.Request(
        url, data=base64.b64encode(frame),
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=45).read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:120].decode('utf-8', 'replace')}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main() -> None:
    key = os.getenv("ROBOFLOW_API_KEY", "")
    if not key:
        raise SystemExit("set ROBOFLOW_API_KEY first")

    raw = get(FRAME_URL)
    frame = upscale(raw)
    from PIL import Image
    sent_w, sent_h = Image.open(io.BytesIO(frame)).size
    print(f"live frame {Image.open(io.BytesIO(raw)).size} -> sending {sent_w}x{sent_h}\n")

    winner = None
    for host in HOSTS:
        for model in CANDIDATES:
            body, err = try_model(host, model, frame, key)
            label = f"{host.split('//')[1].split('.')[0]:11} {model:14}"
            if err:
                print(f"  {label} FAIL  {err}")
                continue

            preds = body.get("predictions", [])
            classes = {str(p.get("class", "")).lower() for p in preds}
            veh = sorted(classes & VEHICLES)
            ped = sorted(classes & PEOPLE)
            img = body.get("image", {})
            n_ped = sum(1 for p in preds
                        if str(p.get("class", "")).lower() in PEOPLE)
            n_veh = sum(1 for p in preds
                        if str(p.get("class", "")).lower() in VEHICLES)

            ok = bool(veh and ped)
            print(f"  {label} {'OK  ' if ok else 'PART'}  "
                  f"{len(preds):>3} preds  {n_ped} people  {n_veh} vehicles  "
                  f"image={img.get('width')}x{img.get('height')}")
            if not ok:
                print(f"{'':>28}classes seen: {sorted(classes)[:8]}")

            # The coordinate frame is the thing that silently corrupts TTC.
            if img.get("width") and int(img["width"]) != sent_w:
                print(f"{'':>28}WARNING: reported width {img['width']} != "
                      f"sent {sent_w}. Ground projection would be wrong.")

            if ok and winner is None:
                winner = (host, model, n_ped, n_veh)

    print()
    if winner:
        host, model, n_ped, n_veh = winner
        print(f"USE THIS: {model}  ({n_ped} people, {n_veh} vehicles detected)")
        print("\nPaste into your deploy:\n")
        print(f'  --set-env-vars "ROBOFLOW_API_KEY={key[:4]}...,'
              f'ROBOFLOW_MODEL={model},ROBOFLOW_URL={host},'
              f'WEBHOOK_URL=https://ghost-v2x-receiver-73791867861.us-east1.run.app/webhook"')
    else:
        print("No candidate returned both a person and a vehicle class.")
        print("Present the /api/replay path; it needs no detector at all.")


if __name__ == "__main__":
    main()
