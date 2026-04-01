#!/bin/bash
# One-command deploy script for MIST API on Cloud Run using Terraform
# Uses Cloud Build for container builds (no local Docker required)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MIST API Cloud Run Deploy Script ===${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found${NC}"
    exit 1
fi

if ! command -v terraform &> /dev/null; then
    # Try to find terraform in common locations
    if [ -f "$HOME/.local/bin/terraform" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo -e "${RED}Error: Terraform not found${NC}"
        echo "Please install Terraform: https://developer.hashicorp.com/terraform/downloads"
        exit 1
    fi
fi

echo -e "${GREEN}✓ All prerequisites found${NC}"
echo ""

# Get project ID
echo -e "${YELLOW}Configuration${NC}"
read -p "Enter your GCP Project ID: " PROJECT_ID
read -p "Enter your GCP Region [us-west1]: " REGION
REGION=${REGION:-us-west1}

export PROJECT_ID
export REGION

# Set gcloud project
echo -e "${YELLOW}Setting gcloud project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com secretmanager.googleapis.com cloudbuild.googleapis.com containerregistry.googleapis.com iam.googleapis.com --quiet

# Build with Cloud Build (no local Docker needed)
echo -e "${YELLOW}Building with Cloud Build...${NC}"
cd "$(dirname "$0")"

IMAGE_TAG="gcr.io/${PROJECT_ID}/mist-api:latest"

gcloud builds submit --tag ${IMAGE_TAG} .

echo -e "${GREEN}✓ Image built and pushed: ${IMAGE_TAG}${NC}"
echo ""

# Create terraform.tfvars if it doesn't exist
cd terraform

if [ ! -f terraform.tfvars ]; then
    echo -e "${YELLOW}Creating terraform.tfvars from environment variables...${NC}"
    : "${DATABASE_URL:?Set DATABASE_URL (Postgres connection string for Secret Manager)}"
    : "${SAMBANOVA_API_KEY:?Set SAMBANOVA_API_KEY}"
    : "${CHROMA_DB_API_KEY:?Set CHROMA_DB_API_KEY}"
    : "${CHROMA_DB_TENANT:?Set CHROMA_DB_TENANT}"
    cat > terraform.tfvars << EOF
project_id        = "${PROJECT_ID}"
region            = "${REGION}"
service_name      = "mist-api"
container_image   = "${IMAGE_TAG}"

# Written locally only; Terraform stores values in Google Secret Manager.
database_url      = "${DATABASE_URL}"
sambanova_api_key = "${SAMBANOVA_API_KEY}"
chromadb_api_key  = "${CHROMA_DB_API_KEY}"
chromadb_tenant   = "${CHROMA_DB_TENANT}"
EOF
    echo -e "${GREEN}✓ Created terraform.tfvars (file is gitignored — do not commit)${NC}"
else
    echo -e "${YELLOW}terraform.tfvars already exists, using existing configuration${NC}"
fi

# Initialize Terraform
echo -e "${YELLOW}Initializing Terraform...${NC}"
terraform init

# Plan deployment
echo -e "${YELLOW}Planning deployment...${NC}"
terraform plan -out=tfplan

# Apply deployment
echo ""
echo -e "${YELLOW}Ready to deploy!${NC}"
read -p "Do you want to apply the deployment? (yes/no): " CONFIRM

if [ "$CONFIRM" = "yes" ]; then
    echo -e "${YELLOW}Applying deployment...${NC}"
    terraform apply tfplan
    
    # Get service URL
    SERVICE_URL=$(terraform output -raw service_url)
    
    echo ""
    echo -e "${GREEN}=== Deployment Complete! ===${NC}"
    echo ""
    echo -e "Service URL: ${GREEN}${SERVICE_URL}${NC}"
    echo ""
    echo "Test the API:"
    echo "  curl ${SERVICE_URL}/health"
    echo ""
    echo "View logs:"
    echo "  gcloud logging read 'resource.type=cloud_run_revision' --limit=50"
    echo ""
    echo "Manage service:"
    echo "  gcloud run services describe mist-api --region ${REGION}"
else
    echo -e "${YELLOW}Deployment cancelled${NC}"
    exit 0
fi
