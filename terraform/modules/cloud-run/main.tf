# Cloud Run module for deploying MIST API

variable "project_id" {
  type = string
}

variable "region" {
  type = string
}

variable "service_name" {
  type = string
}

variable "container_image" {
  type = string
}

variable "secret_refs" {
  type = map(object({
    secret_name = string
    version     = string
  }))
  sensitive = true
}

variable "memory" {
  type    = string
  default = "6Gi"
}

variable "cpu" {
  type    = string
  default = "1"
}

variable "max_instances" {
  type    = number
  default = 3
}

variable "min_instances" {
  type    = number
  default = 0
}

variable "service_account_email" {
  type    = string
  default = ""
}

variable "google_oauth_client_ids" {
  type    = string
  default = ""
}

# Cloud Run service
resource "google_cloud_run_service" "mist_api" {
  name     = var.service_name
  location = var.region
  
  template {
    spec {
      container_concurrency = 80
      timeout_seconds       = 300
      service_account_name  = var.service_account_email
      
      containers {
        image = var.container_image
        
        ports {
          container_port = 8000
        }
        
        resources {
          limits = {
            memory = var.memory
            cpu    = var.cpu
          }
        }
        
        # Environment variables from Secret Manager
        env {
          name = "DATABASE_URL"
          value_from {
            secret_key_ref {
              name = "mist-database-url"
              key  = "latest"
            }
          }
        }
        env {
          name = "SAMBANOVA_API_KEY"
          value_from {
            secret_key_ref {
              name = "mist-sambanova-api-key"
              key  = "latest"
            }
          }
        }
        env {
          name = "CHROMA_DB_API_KEY"
          value_from {
            secret_key_ref {
              name = "mist-chromadb-api-key"
              key  = "latest"
            }
          }
        }
        env {
          name = "CHROMA_DB_TENANT"
          value_from {
            secret_key_ref {
              name = "mist-chromadb-tenant"
              key  = "latest"
            }
          }
        }
        env {
          name = "NEBIUS_API_KEY"
          value_from {
            secret_key_ref {
              name = "nebius-api-key"
              key  = "latest"
            }
          }
        }
        env {
          name = "GEMINI_API_KEY"
          value_from {
            secret_key_ref {
              name = "mist-gemini-api-key"
              key  = "latest"
            }
          }
        }
        env {
          name = "COHERE_API_KEY"
          value_from {
            secret_key_ref {
              name = "mist-cohere-api-key"
              key  = "latest"
            }
          }
        }
        env {
          name = "OPENAI_API_KEY"
          value_from {
            secret_key_ref {
              name = "mist-openai-api-key"
              key  = "latest"
            }
          }
        }
        env {
          name = "REDIS_URL"
          value_from {
            secret_key_ref {
              name = "mist-redis-url"
              key  = "latest"
            }
          }
        }
        env {
          name = "GOOGLE_OAUTH_CLIENT_IDS"
          value = var.google_oauth_client_ids
        }
        
        # Startup probe
        startup_probe {
          http_get {
            path = "/health"
            port = 8000
          }
          initial_delay_seconds = 10
          period_seconds        = 5
          failure_threshold     = 6
        }
        
        # Liveness probe
        liveness_probe {
          http_get {
            path = "/health"
            port = 8000
          }
          period_seconds    = 10
          failure_threshold = 3
        }
      }
    }
    
    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = var.min_instances
        "autoscaling.knative.dev/maxScale" = var.max_instances
        "run.googleapis.com/cpu-throttling" = var.min_instances == 0 ? "true" : "false"
      }
    }
  }
  
  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Allow unauthenticated access (for public API)
resource "google_cloud_run_service_iam_member" "public" {
  service  = google_cloud_run_service.mist_api.name
  location = google_cloud_run_service.mist_api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Output service URL
output "service_url" {
  value = google_cloud_run_service.mist_api.status[0].url
}
