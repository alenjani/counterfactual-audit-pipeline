#!/usr/bin/env bash
# Spin up an L4 VM, run the pipeline inside the container, store outputs in GCS, auto-stop.
# Usage: ./gcp/run_on_l4.sh configs/mvp.yaml
#
# Required env:
#   GCP_PROJECT_ID, GCS_BUCKET, GCP_REGION (default us-central1), GCP_ZONE (default us-central1-a)
set -euo pipefail

CONFIG="${1:-configs/mvp.yaml}"
: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
: "${GCS_BUCKET:?Set GCS_BUCKET (e.g. cap-pipeline-runs)}"
GCP_REGION="${GCP_REGION:-us-central1}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-cap-l4-$(date +%Y%m%d-%H%M%S)}"
IMAGE="${IMAGE:-${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/cap/cap:latest}"

cat > /tmp/cap_startup.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
# Install NVIDIA + Docker GPU runtime
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

docker pull "${IMAGE}"
docker run --rm --gpus all \
  -e GCP_PROJECT_ID="${GCP_PROJECT_ID}" \
  -e GCS_BUCKET="${GCS_BUCKET}" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  "${IMAGE}" bash -lc "bash scripts/run_pipeline.sh ${CONFIG}"

# Auto-delete the VM on completion
gcloud compute instances delete "${INSTANCE_NAME}" --zone="${GCP_ZONE}" --quiet
EOF

echo ">>> Creating L4 VM: ${INSTANCE_NAME}"
gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${GCP_PROJECT_ID}" \
  --zone="${GCP_ZONE}" \
  --machine-type=g2-standard-8 \
  --accelerator="type=nvidia-l4,count=1" \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --metadata-from-file=startup-script=/tmp/cap_startup.sh \
  --scopes=https://www.googleapis.com/auth/cloud-platform

echo ">>> VM created. Tail startup logs:"
echo "  gcloud compute instances get-serial-port-output ${INSTANCE_NAME} --zone=${GCP_ZONE}"
echo ">>> The VM will auto-delete itself when the pipeline finishes."
