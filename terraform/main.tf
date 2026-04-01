# Terraform configuration for MIST API on GCP
# Uses Cloud Run (free tier eligible) + Secret Manager

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
    "containerregistry.googleapis.com",
    "iam.googleapis.com",
  ])
  
  service = each.value
  disable_on_destroy = false
}

# Module: IAM (Service Account)
module "iam" {
  source = "./modules/iam"
  
  project_id = var.project_id
  
  depends_on = [google_project_service.apis]
}

# Module: Secret Manager
module "secrets" {
  source = "./modules/secret-manager"
  
  secrets = {
    database-url      = var.database_url
    sambanova-api-key = var.sambanova_api_key
    chromadb-api-key  = var.chromadb_api_key
    chromadb-tenant   = var.chromadb_tenant
    redis-url         = var.redis_url
  }
  
  service_account_email = module.iam.service_account_email
  
  depends_on = [module.iam]
}

# Module: Cloud Run
module "cloud_run" {
  source = "./modules/cloud-run"
  
  project_id   = var.project_id
  region       = var.region
  service_name = var.service_name
  
  # Use existing image or build new one
  container_image = var.container_image
  
  # Secrets from Secret Manager
  secret_refs = module.secrets.secret_refs
  
  # Service account for minimal permissions
  service_account_email = module.iam.service_account_email
  
  # Resource limits (2Gi needed for Qwen3 embeddings)
  memory = "2Gi"
  cpu    = "1"
  
  # Concurrency and scaling
  max_instances = 3
  min_instances = 0  # Scale to 0 when idle (cost savings)
  
  depends_on = [module.iam, module.secrets]
}

# Output the service URL
output "service_url" {
  value = module.cloud_run.service_url
}
