# Signal controller (mock)

The downstream actuator for Ghost-V2X. Stands in for the thing that would
actually act on an alert: a traffic signal controller, a crosswalk LED driver,
or an operations desk.

It exists so the loop closes on real infrastructure instead of ending at a log
line. Deployed separately from the sensor, and the sensor POSTs to it.

## Deliberately dumb

It does not compute risk, does not second-guess the recommended action, and
does not filter. Ghost-V2X owns that reasoning.

An actuator that re-litigates upstream decisions is how two systems end up
disagreeing about whether to hold a signal — and at an intersection, that
disagreement is the failure.

## Endpoints

| Route | Purpose |
|---|---|
| `POST /webhook` | Receives an alert, returns the action taken |
| `/` | Live feed: risk, action, rationale, TTC, miss distance |
| `/api/alerts` | Same data as JSON |
| `/health` | Liveness (`/healthz` is reserved by Google's frontend) |

State is in-memory and bounded to 25 alerts. This is a demonstration actuator,
not a record of account — the durable conflict log lives in the sensor.

## Deploy

```bash
cd receiver
gcloud run deploy ghost-v2x-receiver --source . --region us-east1 \
  --allow-unauthenticated --port 8080 --memory 512Mi \
  --min-instances 1 --max-instances 1
```

`--max-instances 1` matters: alerts are held in memory per instance, so
scaling past one would scatter them and the feed would show a random subset
depending on which instance answered.
