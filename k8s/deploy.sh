#!/bin/bash
# Build and deploy MIST API to GKE

set -e

PROJECT_ID="your-gcp-project-id"  # Change this
REGION="us-west1"
CLUSTER_NAME="mist-cluster"
IMAGE_NAME="gcr.io/${PROJECT_ID}/mist-api"

echo "=== Building MIST API Docker Image ==="

# Build Docker image
docker build -t ${IMAGE_NAME}:latest .

# Push to GCR
docker push ${IMAGE_NAME}:latest

echo "=== Deploying to GKE ==="

# Get GKE credentials
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION}

# Apply secrets first (base64 encoded)
kubectl apply -f k8s/secrets.yaml

# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Wait for rollout
echo "Waiting for deployment to complete..."
kubectl rollout status deployment/mist-api

echo "=== Deployment Complete ==="
echo ""
echo "Check status:"
echo "  kubectl get pods"
echo "  kubectl get svc"
echo "  kubectl get ingress"
echo ""
echo "View logs:"
echo "  kubectl logs -f deployment/mist-api"
