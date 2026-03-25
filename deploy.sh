#!/bin/bash
# Deploy Video Agent API to Cloud Run
# Usage: ./deploy.sh

set -e

PROJECT_ID="volkano-ai"
REGION="us-central1"
SERVICE_NAME="video-agent-api"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "==> Building container image..."
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}"

echo "==> Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars "\
SERPAPI_API_KEY=${SERPAPI_API_KEY},\
GOOGLE_API_KEY=${GOOGLE_API_KEY},\
GOOGLE_CLOUD_PROJECT=${PROJECT_ID},\
VERTEXAI_PROJECT=${PROJECT_ID},\
VERTEXAI_LOCATION=global,\
GOOGLE_GENAI_USE_VERTEXAI=FALSE"

echo "==> Done! Service URL:"
gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format "value(status.url)"
