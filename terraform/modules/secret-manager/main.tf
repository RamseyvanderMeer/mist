# Secret Manager module for storing sensitive configuration

variable "secrets" {
  description = "Map of secret names to values"
  type        = map(string)
  sensitive   = true
}

variable "service_account_email" {
  description = "Service account email to grant access"
  type        = string
}

# Create secrets in Secret Manager (use non-sensitive keys)
resource "google_secret_manager_secret" "secrets" {
  for_each = toset(["database-url", "sambanova-api-key", "chromadb-api-key", "chromadb-tenant", "redis-url"])
  
  secret_id = "mist-${each.value}"
  
  replication {
    auto {}
  }
}

# Create secret versions
resource "google_secret_manager_secret_version" "versions" {
  for_each = toset(["database-url", "sambanova-api-key", "chromadb-api-key", "chromadb-tenant", "redis-url"])
  
  secret      = google_secret_manager_secret.secrets[each.value].id
  secret_data = var.secrets[each.value]
}

# Grant service account access to secrets
resource "google_secret_manager_secret_iam_member" "secret_access" {
  for_each = toset(["database-url", "sambanova-api-key", "chromadb-api-key", "chromadb-tenant", "redis-url"])
  
  secret_id = google_secret_manager_secret.secrets[each.value].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.service_account_email}"
}

# Output secret references for Cloud Run
output "secret_refs" {
  value = {
    for key in ["database-url", "sambanova-api-key", "chromadb-api-key", "chromadb-tenant", "redis-url"] : key => {
      secret_name = "mist-${key}"
      version     = "latest"
    }
  }
  sensitive = true
}
