#!/usr/bin/env bash
#
# teardown.sh — Remove all AWS resources created by deploy.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

# Load config
CONFIG_FILE="$SCRIPT_DIR/config.env"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo -e "${RED}No config.env found. Nothing to tear down.${NC}"
    exit 1
fi

AWS_REGION="${AWS_REGION:-us-east-1}"
ENVIRONMENT="${ENVIRONMENT:-rasyn-prod}"
ECR_REPO_NAME="${ECR_REPO_NAME:-rasyn}"
S3_BUCKET="${S3_BUCKET:-}"

echo -e "${RED}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║         TEARDOWN — This will DESTROY:                ║${NC}"
echo -e "${RED}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  - CloudFormation stack: $ENVIRONMENT"
echo "    (EC2 instance, ALB, security groups, IAM roles)"
echo "  - ECR repository:       $ECR_REPO_NAME"
echo "  - S3 bucket:            $S3_BUCKET (optional)"
echo ""
read -rp "Type 'destroy' to confirm: " confirm
if [[ "$confirm" != "destroy" ]]; then
    echo "Aborted."
    exit 0
fi

# 1. Delete CloudFormation stack
echo -e "\n${YELLOW}Deleting CloudFormation stack '$ENVIRONMENT'...${NC}"
if aws cloudformation describe-stacks --stack-name "$ENVIRONMENT" --region "$AWS_REGION" &>/dev/null; then
    aws cloudformation delete-stack --stack-name "$ENVIRONMENT" --region "$AWS_REGION"
    echo "Waiting for stack deletion..."
    aws cloudformation wait stack-delete-complete --stack-name "$ENVIRONMENT" --region "$AWS_REGION"
    echo -e "${GREEN}Stack deleted.${NC}"
else
    echo "Stack not found — skipping."
fi

# 2. Delete ECR repository
echo -e "\n${YELLOW}Deleting ECR repository '$ECR_REPO_NAME'...${NC}"
if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" &>/dev/null; then
    aws ecr delete-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION" --force > /dev/null
    echo -e "${GREEN}ECR repository deleted.${NC}"
else
    echo "ECR repository not found — skipping."
fi

# 3. Optionally delete S3 bucket
if [[ -n "$S3_BUCKET" ]]; then
    echo ""
    read -rp "Also delete S3 bucket '$S3_BUCKET' and ALL model files? (yes/no): " del_s3
    if [[ "$del_s3" == "yes" ]]; then
        echo -e "${YELLOW}Deleting S3 bucket...${NC}"
        aws s3 rb "s3://$S3_BUCKET" --force --region "$AWS_REGION"
        echo -e "${GREEN}S3 bucket deleted.${NC}"
    else
        echo "S3 bucket preserved."
    fi
fi

echo -e "\n${GREEN}Teardown complete. All AWS resources removed.${NC}"
