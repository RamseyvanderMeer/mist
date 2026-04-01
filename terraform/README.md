# Terraform deployment for MIST API

## Prerequisites

1. GCP project with billing enabled
2. Terraform installed (>= 1.0)
3. gcloud CLI authenticated

## Setup

### 1. Configure Variables

Create `terraform.tfvars`:

```hcl
project_id = "your-gcp-project-id"
region     = "us-west1"

# Secrets (will be stored in Secret Manager)
database_url      = "postgresql://..."
sambanova_api_key = "your-key"
chromadb_api_key  = "your-key"
chromadb_tenant   = "your-tenant"
```

### 2. Initialize Terraform

```bash
cd terraform
terraform init
```

### 3. Plan Deployment

```bash
terraform plan
```

### 4. Deploy

```bash
terraform apply
```

## Free Tier Eligibility

**Cloud Run:**
- ✅ 2 million requests/month free
- ✅ 360,000 GB-seconds of memory free
- ✅ 180,000 vCPU-seconds free
- ✅ 1 GB egress free

**Secret Manager:**
- ✅ 6 active secret versions free
- ✅ 10,000 access operations free

**Estimated Cost for Low Traffic:**
- **$0-5/month** (within free tier)
- If you exceed: ~$0.40 per million requests

## Architecture

```
User Request
    ↓
Cloud Run (auto-scaling, 0-N instances)
    ↓
FastAPI Application
    ↓
Secret Manager (secure config)
    ↓
External APIs (SambaNova, ChromaDB, Neon)
```

## Managing Secrets

### View Secrets
```bash
gcloud secrets list
```

### Update a Secret
```bash
echo -n "new-value" | gcloud secrets versions add mist-database-url --data-file=-
```

### Rotate Secrets
1. Add new version to Secret Manager
2. Trigger Cloud Run revision (redeploy)
3. Remove old version

## Scaling

Cloud Run automatically scales:
- **0 instances** when idle (no cost)
- **Up to 3 instances** under load (configurable)

Manual scaling:
```bash
gcloud run services update mist-api --min-instances=1
```

## Monitoring

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=mist-api"

# View metrics
gcloud monitoring metrics list | grep run
```

## Cleanup

```bash
terraform destroy
```

This will remove all resources including secrets.
