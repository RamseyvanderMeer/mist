# Variables for MIST API Terraform configuration

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-west1"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "mist-api"
}

variable "container_image" {
  description = "Container image URL (gcr.io/PROJECT_ID/IMAGE:TAG)"
  type        = string
  default     = "gcr.io/PROJECT_ID/mist-api:latest"
}

# Secrets (marked as sensitive)
variable "database_url" {
  description = "PostgreSQL database URL"
  type        = string
  sensitive   = true
}

variable "sambanova_api_key" {
  description = "SambaNova API key"
  type        = string
  sensitive   = true
}

variable "chromadb_api_key" {
  description = "ChromaDB API key"
  type        = string
  sensitive   = true
}

variable "chromadb_tenant" {
  description = "ChromaDB tenant ID"
  type        = string
  sensitive   = true
}

variable "redis_url" {
  description = "Redis URL for rate limiting"
  type        = string
  sensitive   = true
}

variable "google_oauth_client_ids" {
  description = "Comma-separated list of Google OAuth client IDs for web/mobile app auth"
  type        = string
  default     = ""
}

variable "guest_cookie_secret" {
  description = "Secret used to sign guest cookies"
  type        = string
  sensitive   = true
}

variable "allowed_origins" {
  description = "Comma-separated allowed origins for credentialed frontend access"
  type        = string
  default     = "https://mist-expo.vercel.app,http://localhost:8081"
}
