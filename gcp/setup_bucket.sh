#!/usr/bin/env bash
# One-time: create the GCS bucket for runs + datasets.
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCS_BUCKET:?Set GCS_BUCKET}"
GCP_REGION="${GCP_REGION:-us-central1}"

gcloud storage buckets describe "gs://${GCS_BUCKET}" >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://${GCS_BUCKET}" \
    --project="${GCP_PROJECT_ID}" \
    --location="${GCP_REGION}" \
    --uniform-bucket-level-access

# Suggested layout
gcloud storage cp /dev/null "gs://${GCS_BUCKET}/data/.placeholder"
gcloud storage cp /dev/null "gs://${GCS_BUCKET}/runs/.placeholder"

echo ">>> Bucket ready: gs://${GCS_BUCKET}"
