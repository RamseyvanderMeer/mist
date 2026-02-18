#!/bin/bash
# Manual cleanup script - removes Cloud Run Job, Cloud SQL instances, and optionally GCS buckets.
# Use when deploy_and_run.sh was interrupted or for ad-hoc teardown.
set -e

PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
REGION="${REGION:-us-central1}"
JOB_NAME="mist-scraper-job"

if [ -z "$PROJECT_ID" ]; then
  echo "Error: gcloud project not set."
  exit 1
fi

echo "Cleaning up MIST scraper resources (project=$PROJECT_ID)..."

# Delete Cloud Run Job
if gcloud run jobs describe "$JOB_NAME" --region="$REGION" --project="$PROJECT_ID" &>/dev/null; then
  echo "Deleting Cloud Run Job: $JOB_NAME"
  gcloud run jobs delete "$JOB_NAME" --region="$REGION" --project="$PROJECT_ID" --quiet
else
  echo "Job $JOB_NAME not found, skipping."
fi

# Delete Cloud SQL instances matching pattern
echo "Listing Cloud SQL instances..."
for INST in $(gcloud sql instances list --project="$PROJECT_ID" --format="value(name)" 2>/dev/null | grep "mist-scraper-db-" || true); do
  echo "Deleting Cloud SQL instance: $INST"
  gcloud sql instances delete "$INST" --project="$PROJECT_ID" --quiet
done

# Optional: delete GCS buckets (uncomment and set BUCKET_NAME if needed)
# BUCKET_NAME="mist-data-XXXX"
# gcloud storage rm -r "gs://${BUCKET_NAME}" 2>/dev/null || true

echo "Cleanup complete."
