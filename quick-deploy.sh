#!/bin/bash
# Quick deploy script using gcloud commands

set -e

echo "=== Deploying MIST API to Cloud Run ==="
echo ""

# Build and push with Cloud Build
echo "Building container..."
gcloud builds submit --tag gcr.io/mist-487607/mist-api:latest .

# Deploy to Cloud Run with proper settings
echo "Deploying to Cloud Run..."
gcloud run deploy mist-api \
  --image gcr.io/mist-487607/mist-api:latest \
  --region us-west1 \
  --platform managed \
  --memory 2Gi \
  --cpu 1 \
  --concurrency 80 \
  --max-instances 3 \
  --min-instances 0 \
  --timeout 300 \
  --service-account mist-api-sa@mist-487607.iam.gserviceaccount.com \
  --set-secrets DATABASE_URL=mist-database-url:latest,SAMBANOVA_API_KEY=mist-sambanova-api-key:latest,CHROMADB_API_KEY=mist-chromadb-api-key:latest,CHROMADB_TENANT=mist-chromadb-tenant:latest \
  --allow-unauthenticated

echo ""
echo "=== Deployment Complete ==="
gcloud run services describe mist-api --region us-west1 --format 'value(status.url)'
