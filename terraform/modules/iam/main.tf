# GCP Service Account for MIST API
# Minimal permissions following principle of least privilege

resource "google_service_account" "mist_api" {
  account_id   = "mist-api-sa"
  display_name = "MIST API Service Account"
  description  = "Service account for MIST API Cloud Run service"
}

# Note: Secret access is granted in the secret-manager module
# to avoid dependency issues

# Grant Cloud Logging access
resource "google_project_iam_member" "logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.mist_api.email}"
}

# Grant Cloud Monitoring access
resource "google_project_iam_member" "monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.mist_api.email}"
}

# Grant Cloud Trace access (for distributed tracing)
resource "google_project_iam_member" "cloudtrace" {
  project = var.project_id
  role    = "roles/cloudtrace.agent"
  member  = "serviceAccount:${google_service_account.mist_api.email}"
}

output "service_account_email" {
  value = google_service_account.mist_api.email
}
