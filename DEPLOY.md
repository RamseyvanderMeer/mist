# Quick Start: Deploy MIST API to Cloud Run

## One-Command Deploy

```bash
./deploy.sh
```

This script will:
1. ✅ Check prerequisites (gcloud, terraform, docker)
2. ✅ Enable required GCP APIs
3. ✅ Build and push Docker image to GCR
4. ✅ Create terraform.tfvars with your configuration
5. ✅ Initialize and plan Terraform deployment
6. ✅ Deploy to Cloud Run
7. ✅ Output the service URL

## Manual Steps (if you prefer)

### 1. Prerequisites

```bash
# Install gcloud
https://cloud.google.com/sdk/docs/install

# Install Terraform
https://developer.hashicorp.com/terraform/downloads

# Authenticate
gcloud auth login
gcloud auth application-default login
```

### 2. Build and Push Image

```bash
# Set project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Build
docker build -t gcr.io/${PROJECT_ID}/mist-api:latest .
docker push gcr.io/${PROJECT_ID}/mist-api:latest
```

### 3. Deploy with Terraform

```bash
cd terraform

# Create terraform.tfvars
cat > terraform.tfvars << EOF
project_id        = "your-project-id"
region            = "us-west1"
container_image   = "gcr.io/your-project-id/mist-api:latest"
database_url      = "your-database-url"
sambanova_api_key = "your-key"
chromadb_api_key  = "your-key"
chromadb_tenant   = "your-tenant"
EOF

# Deploy
terraform init
terraform plan
terraform apply
```

## Verify Deployment

```bash
# Get service URL
export SERVICE_URL=$(cd terraform && terraform output -raw service_url)

# Test health endpoint
curl ${SERVICE_URL}/health

# Expected response:
# {"status": "healthy", "service": "MIST API"}
```

## Update Deployment

```bash
# Make code changes, then:
./deploy.sh
# or manually:
docker build -t gcr.io/${PROJECT_ID}/mist-api:latest .
docker push gcr.io/${PROJECT_ID}/mist-api:latest
cd terraform && terraform apply
```

## View Logs

```bash
# Real-time logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=mist-api"

# Recent logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=mist-api" --limit=50
```

## Cleanup

```bash
cd terraform
terraform destroy
```

## Troubleshooting

**Issue: Image pull error**
```bash
# Ensure GCR is accessible
gcloud auth configure-docker
```

**Issue: Secret not found**
```bash
# Check secrets exist
gcloud secrets list
```

**Issue: Service not accessible**
```bash
# Check service status
gcloud run services describe mist-api --region us-west1
```

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTPS
       ▼
┌─────────────────┐
│  Cloud Load     │
│   Balancer      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Cloud Run     │────▶│  Secret Manager │
│   (FastAPI)     │     │   (Config)      │
└────────┬────────┘     └─────────────────┘
         │
         ├──▶ Neon PostgreSQL (Database)
         ├──▶ SambaNova API (Embeddings)
         ├──▶ ChromaDB Cloud (Vector Store)
         └──▶ BMWFault DB (P-code mappings)
```

## Cost Estimate

| Resource | Free Tier | Paid (if exceeded) |
|----------|-----------|-------------------|
| Cloud Run | 2M requests/mo | $0.40/million |
| Secret Manager | 6 versions | $0.06/version/mo |
| **Total** | **$0** | **~$0-5/mo** |

## Security

- ✅ Secrets stored in Google Secret Manager (encrypted at rest)
- ✅ Cloud Run uses minimal service account
- ✅ HTTPS only (automatic TLS)
- ✅ No secrets in code or containers
