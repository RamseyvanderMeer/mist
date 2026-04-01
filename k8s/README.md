# GCP Kubernetes Deployment Guide for MIST API

## Prerequisites

1. GCP project with billing enabled
2. gcloud CLI installed and authenticated
3. Docker installed
4. kubectl installed

## Setup Steps

### 1. Create GKE Cluster (if not exists)

```bash
# Set project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Create cluster (e2-medium = 2 vCPU, 4GB RAM)
gcloud container clusters create mist-cluster \
  --region us-west1 \
  --machine-type e2-medium \
  --num-nodes 2 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 3 \
  --enable-autorepair
```

### 2. Configure Secrets

Edit `k8s/secrets.yaml` and replace the `REPLACE_WITH_*` placeholders with real values (this file must not contain committed secrets).

**Important:** In production, use Google Secret Manager instead:

```bash
# Create secrets in Secret Manager
gcloud secrets create mist-database-url --data-file=-
gcloud secrets create mist-sambanova-key --data-file=-
# etc.
```

### 3. Build and Deploy

```bash
chmod +x k8s/deploy.sh
./k8s/deploy.sh
```

### 4. Configure Ingress (Optional - for external access)

```bash
# Reserve static IP
gcloud compute addresses create mist-api-ip --global

# Create managed SSL certificate
cat <<EOF | kubectl apply -f -
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: mist-api-cert
spec:
  domains:
    - api.yourdomain.com
EOF

# Update deployment.yaml with your domain
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods

# Check logs
kubectl logs -f deployment/mist-api

# Test health endpoint
kubectl port-forward svc/mist-api-service 8080:80
curl http://localhost:8080/health
```

## Cost Estimate (Low Traffic)

- **GKE Cluster**: ~$75/month (e2-medium x 2 nodes)
- **Load Balancer**: ~$18/month
- **Cloud SQL** (if needed): ~$7/month (db-f1-micro)
- **Total**: ~$100/month

**For true MVP, you can reduce to:**
- 1 node (not recommended for production)
- Use Cloud Run instead (pay per request)

## Scaling

```bash
# Manual scale
kubectl scale deployment mist-api --replicas=3

# Or enable HPA (Horizontal Pod Autoscaler)
kubectl autoscale deployment mist-api --cpu-percent=70 --min=2 --max=5
```

## Monitoring

```bash
# View metrics
kubectl top pods

# View logs
kubectl logs -f deployment/mist-api --tail=100

# Describe pod for debugging
kubectl describe pod <pod-name>
```

## Troubleshooting

1. **Image pull errors**: Ensure GCR is accessible
2. **OOMKilled**: Increase memory limits in deployment.yaml
3. **CrashLoopBackOff**: Check logs with `kubectl logs`
4. **Connection refused**: Ensure service is targeting correct port

## Alternative: Cloud Run (Cheaper for MVP)

For very low traffic, Cloud Run is cheaper:

```bash
gcloud run deploy mist-api \
  --image gcr.io/$PROJECT_ID/mist-api \
  --platform managed \
  --region us-west1 \
  --allow-unauthenticated \
  --set-env-vars "DATABASE_URL=..." \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 80 \
  --max-instances 3
```

Cost: ~$0-10/month for low traffic (pay per request)
