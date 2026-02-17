#!/usr/bin/env bash
#
# update.sh — Push code changes to running EC2 instance without redeploying.
#
# This is for iterating quickly: updates the code via git pull,
# reinstalls requirements if changed, and restarts services.
#
# Usage:
#   ./deploy/update.sh                          # Uses config.env for instance details
#   ./deploy/update.sh --host <ec2-ip>          # Direct SSH to instance
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# Load config
CONFIG_FILE="${SCRIPT_DIR}/config.env"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
fi

# Get instance IP
if [[ "${1:-}" == "--host" && -n "${2:-}" ]]; then
    INSTANCE_IP="$2"
else
    AWS_REGION="${AWS_REGION:-us-east-1}"
    ENVIRONMENT="${ENVIRONMENT:-rasyn-prod}"

    info "Finding EC2 instance for $ENVIRONMENT..."
    INSTANCE_IP=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --filters \
            "Name=tag:Name,Values=${ENVIRONMENT}-gpu" \
            "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null)

    if [[ -z "$INSTANCE_IP" || "$INSTANCE_IP" == "None" ]]; then
        echo "Could not find running instance. Provide IP directly:"
        echo "  ./deploy/update.sh --host <ec2-ip>"
        exit 1
    fi
fi

ok "Instance: $INSTANCE_IP"
info "Updating code and restarting services..."

# Find SSH key
SSH_KEY=""
for key in "${SCRIPT_DIR}/rasyn-key.pem" "$HOME/.ssh/rasyn-key.pem" "$HOME/.ssh/id_rsa"; do
    if [[ -f "$key" ]]; then
        SSH_KEY="$key"
        break
    fi
done

if [[ -z "$SSH_KEY" ]]; then
    warn "No SSH key found. Trying without key..."
    SSH_CMD="ssh -o StrictHostKeyChecking=no ubuntu@$INSTANCE_IP"
else
    SSH_CMD="ssh -o StrictHostKeyChecking=no -i $SSH_KEY ubuntu@$INSTANCE_IP"
fi

$SSH_CMD << 'REMOTE'
set -euo pipefail

APP_DIR=/opt/rasyn/app
cd $APP_DIR

echo "=== Pulling latest code ==="
git pull origin master

echo "=== Checking for new dependencies ==="
source /opt/rasyn/venv/bin/activate
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -3

echo "=== Restarting services ==="
sudo systemctl restart rasyn-api
sleep 3
sudo systemctl restart rasyn-worker

echo "=== Service status ==="
sudo systemctl status rasyn-api --no-pager -l | head -10
sudo systemctl status rasyn-worker --no-pager -l | head -10

echo "=== Update complete ==="
REMOTE

ok "Code updated and services restarted on $INSTANCE_IP"
echo ""
echo -e "  Health check: ${BLUE}http://$INSTANCE_IP:8000/api/v1/health${NC}"
echo -e "  API docs:     ${BLUE}http://$INSTANCE_IP:8000/docs${NC}"
