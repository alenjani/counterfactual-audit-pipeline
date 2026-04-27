#!/usr/bin/env bash
# Build the CAP container and push to GCP Artifact Registry.
# Required env: GCP_PROJECT_ID, GCP_REGION (default us-central1)
set -euo pipefail

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
GCP_REGION="${GCP_REGION:-us-central1}"
REPO="${GCP_REPO:-cap}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO}/cap:${IMAGE_TAG}"

# One-time: create Artifact Registry repo (idempotent)
gcloud artifacts repositories describe "$REPO" --location="$GCP_REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker \
    --location="$GCP_REGION" \
    --description="Counterfactual Audit Pipeline images"

gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

echo ">>> Building $IMAGE"
docker build -f docker/Dockerfile -t "$IMAGE" .

echo ">>> Pushing $IMAGE"
docker push "$IMAGE"

echo ">>> Done. Image: $IMAGE"
