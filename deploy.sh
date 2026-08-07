#!/usr/bin/env bash
# Ghost-V2X → Cloud Run. Run from Cloud Shell in the repo root.
set -euo pipefail

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-ghost-v2x}"
PROJECT="$(gcloud config get-value project 2>/dev/null)"

if [[ -z "$PROJECT" || "$PROJECT" == "(unset)" ]]; then
  echo "No project set. Run:  gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi

echo "project : $PROJECT"
echo "service : $SERVICE ($REGION)"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project "$PROJECT"

gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --min-instances 1 \
  --max-instances 1 \
  --timeout 3600 \
  --set-env-vars "CAMERA_MATCH=${CAMERA_MATCH:-1 Ave @ 42 St},POLL_SECONDS=${POLL_SECONDS:-2.5}"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo
echo "LIVE: $URL"
echo "state: $URL/api/state"
