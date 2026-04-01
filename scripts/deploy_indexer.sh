#!/bin/bash
# Deploy indexer as Cloud Run Job

set -e

PROJECT_ID="mist-487607"
REGION="us-west1"
JOB_NAME="repair-guide-indexer"

echo "Building indexer container..."
gcloud builds submit --tag gcr.io/${PROJECT_ID}/repair-guide-indexer:latest \
  --file=Dockerfile.indexer .

echo "Creating Cloud Run Job..."
gcloud run jobs create ${JOB_NAME} \
  --image gcr.io/${PROJECT_ID}/repair-guide-indexer:latest \
  --region ${REGION} \
  --max-retries 3 \
  --task-timeout 6h \
  --memory 4Gi \
  --cpu 2 \
  --set-secrets NEBIUS_API_KEY=nebius-api-key:latest \
  --set-secrets CHROMA_DB_API_KEY=mist-chromadb-api-key:latest \
  --set-secrets CHROMA_DB_TENANT=mist-chromadb-tenant:latest \
  --set-env-vars OUTPUT_DIM=4096,BATCH_SIZE=50

echo "Job created! To start indexing:"
echo "gcloud run jobs execute ${JOB_NAME} --region ${REGION}"
