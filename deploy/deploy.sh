#!/usr/bin/env bash
#
# deploy.sh — One-command production deployment of Rasyn Chemistry OS to AWS.
#
# Deploys: FastAPI + Celery worker + PostgreSQL 16 + Redis on a single GPU EC2 instance.
# Models are downloaded from S3 at boot time.
#
# Prerequisites:
#   1. AWS CLI v2 installed and configured (`aws configure`)
#   2. An AWS EC2 key pair created (`aws ec2 create-key-pair --key-name rasyn-key ...`)
#
# Usage:
#   ./deploy/deploy.sh               # Interactive — prompts for missing config
#   ./deploy/deploy.sh --config deploy/config.env   # Use saved config
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─────────────────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "\n${GREEN}━━━ Step $1: $2 ━━━${NC}"; }

# ─────────────────────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────────────────────
CONFIG_FILE="${SCRIPT_DIR}/config.env"
if [[ "${1:-}" == "--config" && -n "${2:-}" ]]; then
    CONFIG_FILE="$2"
fi

if [[ -f "$CONFIG_FILE" ]]; then
    info "Loading config from $CONFIG_FILE"
    source "$CONFIG_FILE"
fi

# ─────────────────────────────────────────────────────────────
# Configuration with defaults
# ─────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
ENVIRONMENT="${ENVIRONMENT:-rasyn-prod}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
S3_BUCKET="${S3_BUCKET:-rasyn-models-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo 'ACCOUNT')}"
KEY_PAIR_NAME="${KEY_PAIR_NAME:-}"

# Paths to model files (on local machine or RunPod — user must have these)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"

# ─────────────────────────────────────────────────────────────
# Preflight checks
# ─────────────────────────────────────────────────────────────
step 0 "Preflight checks"

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        error "$1 is not installed. Please install it first."
        exit 1
    fi
    ok "$1 found"
}

check_cmd aws

# Verify AWS credentials
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) || {
    error "AWS credentials not configured. Run: aws configure"
    exit 1
}
ok "AWS Account: $AWS_ACCOUNT_ID (Region: $AWS_REGION)"

# ─────────────────────────────────────────────────────────────
# Interactive prompts for missing config
# ─────────────────────────────────────────────────────────────
if [[ -z "$KEY_PAIR_NAME" ]]; then
    echo ""
    info "Available EC2 key pairs:"
    aws ec2 describe-key-pairs --region "$AWS_REGION" --query 'KeyPairs[].KeyName' --output table 2>/dev/null || true
    echo ""
    read -rp "Enter EC2 key pair name (or 'create' to make one): " KEY_PAIR_NAME
    if [[ "$KEY_PAIR_NAME" == "create" ]]; then
        KEY_PAIR_NAME="rasyn-key"
        info "Creating key pair '$KEY_PAIR_NAME'..."
        aws ec2 create-key-pair --key-name "$KEY_PAIR_NAME" --region "$AWS_REGION" \
            --query 'KeyMaterial' --output text > "${SCRIPT_DIR}/${KEY_PAIR_NAME}.pem"
        chmod 400 "${SCRIPT_DIR}/${KEY_PAIR_NAME}.pem"
        ok "Key pair saved to ${SCRIPT_DIR}/${KEY_PAIR_NAME}.pem"
    fi
fi

# Get VPC and subnets
info "Detecting default VPC..."
VPC_ID=$(aws ec2 describe-vpcs --region "$AWS_REGION" \
    --filters "Name=is-default,Values=true" \
    --query 'Vpcs[0].VpcId' --output text 2>/dev/null)

if [[ "$VPC_ID" == "None" || -z "$VPC_ID" ]]; then
    error "No default VPC found. Please specify VPC_ID in config.env"
    exit 1
fi
ok "VPC: $VPC_ID"

SUBNET_IDS=$(aws ec2 describe-subnets --region "$AWS_REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
    --query 'Subnets[*].SubnetId' --output text | tr '\t' ',')
ok "Subnets: $SUBNET_IDS"

# ─────────────────────────────────────────────────────────────
# Step 1: Create S3 bucket and upload checkpoints
# ─────────────────────────────────────────────────────────────
step 1 "Create S3 bucket and upload model checkpoints"

if aws s3 ls "s3://$S3_BUCKET" &>/dev/null 2>&1; then
    ok "S3 bucket '$S3_BUCKET' already exists"
else
    info "Creating S3 bucket '$S3_BUCKET'..."
    if [[ "$AWS_REGION" == "us-east-1" ]]; then
        aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION"
    else
        aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION" \
            --create-bucket-configuration LocationConstraint="$AWS_REGION"
    fi
    ok "S3 bucket created"
fi

# Upload checkpoints if directory provided
if [[ -n "$CHECKPOINT_DIR" && -d "$CHECKPOINT_DIR" ]]; then
    info "Uploading checkpoints from $CHECKPOINT_DIR ..."
    aws s3 sync "$CHECKPOINT_DIR" "s3://$S3_BUCKET/models/checkpoints/" \
        --exclude "*/checkpoint-*/*" \
        --exclude "*/optimizer.pt" \
        --exclude "*/scheduler.pt" \
        --exclude "*/rng_state.pth" \
        --exclude "*/training_args.bin" \
        --exclude "*/trainer_state.json"
    ok "Checkpoints uploaded"
else
    warn "CHECKPOINT_DIR not set — skipping checkpoint upload."
    warn "Set CHECKPOINT_DIR in config.env or upload manually:"
    warn "  aws s3 sync /path/to/checkpoints s3://$S3_BUCKET/models/checkpoints/"
fi

if [[ -n "$WEIGHTS_DIR" && -d "$WEIGHTS_DIR" ]]; then
    info "Uploading base weights from $WEIGHTS_DIR ..."
    aws s3 sync "$WEIGHTS_DIR" "s3://$S3_BUCKET/models/weights/"
    ok "Weights uploaded"
else
    warn "WEIGHTS_DIR not set — skipping weights upload."
    warn "Set WEIGHTS_DIR in config.env or upload manually:"
    warn "  aws s3 sync /path/to/weights s3://$S3_BUCKET/models/weights/"
fi

# Upload vocab data
VOCAB_DIR="$PROJECT_ROOT/data/vocab"
if [[ -d "$VOCAB_DIR" ]]; then
    info "Uploading vocab data..."
    aws s3 sync "$VOCAB_DIR" "s3://$S3_BUCKET/models/data/vocab/"
    ok "Vocab data uploaded"
fi

info "S3 bucket contents:"
aws s3 ls "s3://$S3_BUCKET/models/" --recursive --human-readable --summarize 2>/dev/null | tail -5

# ─────────────────────────────────────────────────────────────
# Step 2: Find Deep Learning AMI
# ─────────────────────────────────────────────────────────────
step 2 "Find GPU-ready AMI"

AMI_ID=$(aws ec2 describe-images \
    --region "$AWS_REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
        "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text 2>/dev/null)

if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    warn "Could not find Deep Learning AMI. Trying Ubuntu 22.04 base..."
    AMI_ID=$(aws ec2 describe-images \
        --region "$AWS_REGION" \
        --owners 099720109477 \
        --filters \
            "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
        --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
        --output text)
fi

ok "AMI: $AMI_ID"

# ─────────────────────────────────────────────────────────────
# Step 3: Deploy CloudFormation stack
# ─────────────────────────────────────────────────────────────
step 3 "Deploy CloudFormation stack"

STACK_NAME="$ENVIRONMENT"
TEMPLATE="$SCRIPT_DIR/infrastructure.yaml"

info "Creating/updating CloudFormation stack '$STACK_NAME'..."
info "  Instance type:  $INSTANCE_TYPE"
info "  AMI:            $AMI_ID"
info "  S3 bucket:      $S3_BUCKET"
info "  Key pair:       $KEY_PAIR_NAME"
info "  Services:       FastAPI + Celery + PostgreSQL 16 + Redis"

# Check if stack exists
if aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" &>/dev/null; then
    info "Stack exists — updating..."
    OPERATION="update-stack"
else
    info "Creating new stack..."
    OPERATION="create-stack"
fi

aws cloudformation $OPERATION \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --template-body "file://$TEMPLATE" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameters \
        "ParameterKey=EnvironmentName,ParameterValue=$ENVIRONMENT" \
        "ParameterKey=InstanceType,ParameterValue=$INSTANCE_TYPE" \
        "ParameterKey=KeyPairName,ParameterValue=$KEY_PAIR_NAME" \
        "ParameterKey=AmiId,ParameterValue=$AMI_ID" \
        "ParameterKey=S3ModelsBucket,ParameterValue=$S3_BUCKET" \
        "ParameterKey=VpcId,ParameterValue=$VPC_ID" \
        "ParameterKey=SubnetIds,ParameterValue=\"$SUBNET_IDS\"" \
    --output text

info "Waiting for stack to complete (this takes 5-10 minutes)..."
if [[ "$OPERATION" == "create-stack" ]]; then
    aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME" --region "$AWS_REGION"
else
    aws cloudformation wait stack-update-complete --stack-name "$STACK_NAME" --region "$AWS_REGION"
fi

ok "Stack deployed successfully!"

# ─────────────────────────────────────────────────────────────
# Step 4: Get outputs
# ─────────────────────────────────────────────────────────────
step 4 "Deployment complete!"

ALB_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ALBEndpoint`].OutputValue' \
    --output text)

LOG_GROUP=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`LogGroupName`].OutputValue' \
    --output text)

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Rasyn Chemistry OS — Deployment Complete!        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BLUE}Services deployed:${NC}"
echo -e "    FastAPI API server   (port 8000)"
echo -e "    Celery GPU worker    (concurrency=1)"
echo -e "    PostgreSQL 16        (pgvector enabled)"
echo -e "    Redis 7              (pub/sub + task broker)"
echo ""
echo -e "  ${BLUE}Endpoints:${NC}"
echo -e "    API v1:         ${BLUE}$ALB_ENDPOINT/api/v1/health${NC}"
echo -e "    API v2 (jobs):  ${BLUE}$ALB_ENDPOINT/api/v2/plan${NC}"
echo -e "    API Docs:       ${BLUE}$ALB_ENDPOINT/docs${NC}"
echo -e "    Gradio Demo:    ${BLUE}$ALB_ENDPOINT/demo${NC}"
echo ""
echo -e "  ${BLUE}Frontend:${NC}"
echo -e "    Set NEXT_PUBLIC_API_BASE=$ALB_ENDPOINT in demo/.env.local"
echo -e "    Then: cd demo && npm run dev"
echo ""
echo -e "  ${BLUE}Monitoring:${NC}"
echo -e "    CloudWatch Logs: ${BLUE}$LOG_GROUP${NC}"
echo ""
echo -e "  ${YELLOW}Note: The instance needs ~10-15 min to install PostgreSQL,"
echo -e "  download models, and start serving. Check health endpoint.${NC}"
echo ""

# ─────────────────────────────────────────────────────────────
# Save config for future use
# ─────────────────────────────────────────────────────────────
cat > "$SCRIPT_DIR/config.env" <<CONF
# Rasyn Chemistry OS — AWS Deployment Config
# Auto-generated $(date -u +"%Y-%m-%dT%H:%M:%SZ")
AWS_REGION=$AWS_REGION
ENVIRONMENT=$ENVIRONMENT
INSTANCE_TYPE=$INSTANCE_TYPE
S3_BUCKET=$S3_BUCKET
KEY_PAIR_NAME=$KEY_PAIR_NAME
CHECKPOINT_DIR=$CHECKPOINT_DIR
WEIGHTS_DIR=$WEIGHTS_DIR
ALB_ENDPOINT=$ALB_ENDPOINT
CONF

ok "Config saved to $SCRIPT_DIR/config.env"
