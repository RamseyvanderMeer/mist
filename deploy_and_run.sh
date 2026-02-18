#!/bin/bash
# One-off MIST scraper deployment - creates infra, runs job, destroys infra.
# Target: <$5 total cost. Deployable in ~1 hour.
set -e

PROJECT_ID=$(gcloud config get-value project 2>/dev/null || echo "")
REGION="${REGION:-us-central1}"
JOB_NAME="mist-scraper-job"
DB_NAME="mist-scraper-db-$(date +%s)"
BUCKET_NAME="mist-data-$(date +%s)"

if [ -z "$PROJECT_ID" ]; then
  echo "Error: gcloud project not set. Run: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

echo "Starting One-Off Scraper Deployment (project=$PROJECT_ID, region=$REGION)..."

# 0. Ensure Gemini secret exists
if ! gcloud secrets describe gemini-api-key --project="$PROJECT_ID" &>/dev/null; then
  echo "Creating gemini-api-key secret. You will be prompted for the value."
  echo -n "Enter GEMINI_API_KEY: "
  read -s GEMINI_KEY
  echo
  echo -n "$GEMINI_KEY" | gcloud secrets create gemini-api-key --data-file=- --project="$PROJECT_ID"
fi

# 1. Create GCS Bucket
echo "Creating GCS Bucket..."
gcloud storage buckets create "gs://${BUCKET_NAME}" \
  --location="$REGION" \
  --project="$PROJECT_ID"
if [ -f lifecycle.json ]; then
  gcloud storage buckets update "gs://${BUCKET_NAME}" --lifecycle-file=lifecycle.json --project="$PROJECT_ID"
fi

# 2. Create Ephemeral Cloud SQL
echo "Creating Ephemeral Cloud SQL..."
DB_PASS=$(openssl rand -base64 12)
gcloud sql instances create "$DB_NAME" \
  --database-version=POSTGRES_15 \
  --tier=db-g1-small \
  --region="$REGION" \
  --root-password="$DB_PASS" \
  --project="$PROJECT_ID" \
  --async

# 3. Build Container
echo "Building and pushing container..."
gcloud builds submit --tag "gcr.io/${PROJECT_ID}/${JOB_NAME}:latest" . --project="$PROJECT_ID"

# 4. Wait for DB
echo "Waiting for database to be ready..."
for i in $(seq 1 30); do
  STATE=$(gcloud sql instances describe "$DB_NAME" --format="value(state)" --project="$PROJECT_ID" 2>/dev/null || echo "")
  if [ "$STATE" = "RUNNABLE" ]; then
    break
  fi
  echo "  DB state: $STATE (attempt $i/30)"
  sleep 20
done
if [ "$STATE" != "RUNNABLE" ]; then
  echo "Database failed to become RUNNABLE. Cleaning up..."
  gcloud sql instances delete "$DB_NAME" --project="$PROJECT_ID" --quiet 2>/dev/null || true
  exit 1
fi

# Enable pgvector (optional - app can do this on first connect)
echo "Enabling pgvector..."
gcloud sql databases create scraper_state --instance="$DB_NAME" --project="$PROJECT_ID" 2>/dev/null || true

# 5. Deploy Cloud Run Job
echo "Deploying Cloud Run Job..."
gcloud run jobs deploy "$JOB_NAME" \
  --image "gcr.io/${PROJECT_ID}/${JOB_NAME}:latest" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --set-env-vars "DB_HOST=/cloudsql/${PROJECT_ID}:${REGION}:${DB_NAME},DB_USER=postgres,DB_PASS=${DB_PASS},BUCKET=${BUCKET_NAME}" \
  --set-secrets "GEMINI_API_KEY=gemini-api-key:latest" \
  --add-cloudsql-instances "${PROJECT_ID}:${REGION}:${DB_NAME}" \
  --max-retries 0 \
  --task-timeout 4h \
  --memory 4Gi \
  --cpu 2

# 6. Execute
echo "Executing job..."
gcloud run jobs execute "$JOB_NAME" --region "$REGION" --project="$PROJECT_ID" --wait

# 7. Cleanup
echo "Cleaning up infrastructure..."
gcloud sql instances delete "$DB_NAME" --project="$PROJECT_ID" --quiet
gcloud run jobs delete "$JOB_NAME" --region "$REGION" --project="$PROJECT_ID" --quiet

echo "Done! Output data in gs://${BUCKET_NAME} (lifecycle will delete after 1 day if lifecycle.json was applied)"
