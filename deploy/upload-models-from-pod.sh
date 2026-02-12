#!/usr/bin/env bash
#
# upload-models-from-pod.sh — Upload model checkpoints from RunPod to S3.
#
# Run this ON THE RUNPOD POD (not locally) to upload checkpoints directly.
# This is faster than downloading to local machine first.
#
# Prerequisites:
#   pip install awscli
#   aws configure   (enter your access key, secret key, region)
#
# Usage:
#   bash upload-models-from-pod.sh <s3-bucket-name>
#
set -euo pipefail

S3_BUCKET="${1:-}"
if [[ -z "$S3_BUCKET" ]]; then
    echo "Usage: $0 <s3-bucket-name>"
    echo "Example: $0 rasyn-models-123456789"
    exit 1
fi

RASYN_DIR="${RASYN_DIR:-/workspace/rasyn}"

echo "=== Uploading Rasyn model checkpoints to s3://$S3_BUCKET ==="

# LLM (LoRA adapter + tokenizer) — only the final checkpoint, not intermediates
echo "[1/5] LLM adapter (checkpoints/llm/uspto50k_v6/final/)..."
aws s3 sync "$RASYN_DIR/checkpoints/llm/uspto50k_v6/final/" \
    "s3://$S3_BUCKET/models/checkpoints/llm/uspto50k_v6/final/" \
    --exclude "*/checkpoint-*" \
    --exclude "*.md"

# RetroTransformer v2
echo "[2/5] RetroTransformer v2 (checkpoints/retro_v2/...)..."
aws s3 cp "$RASYN_DIR/checkpoints/retro_v2/uspto50k/best/model.pt" \
    "s3://$S3_BUCKET/models/checkpoints/retro_v2/uspto50k/best/model.pt"

# Forward model — only best checkpoint
echo "[3/5] Forward model (checkpoints/forward/...)..."
aws s3 cp "$RASYN_DIR/checkpoints/forward/uspto50k/best_model.pt" \
    "s3://$S3_BUCKET/models/checkpoints/forward/uspto50k/best_model.pt"

# Graph head (if exists)
if [[ -f "$RASYN_DIR/checkpoints/graph_head/best_model.pt" ]]; then
    echo "[4/5] Graph head..."
    aws s3 cp "$RASYN_DIR/checkpoints/graph_head/best_model.pt" \
        "s3://$S3_BUCKET/models/checkpoints/graph_head/best_model.pt"
else
    echo "[4/5] Graph head — not found, skipping"
fi

# RSGPT base weights (~6.1 GB)
echo "[5/5] RSGPT base weights (weights/rsgpt/finetune_50k.pth) — ~6.1 GB, may take a few minutes..."
aws s3 cp "$RASYN_DIR/weights/rsgpt/finetune_50k.pth" \
    "s3://$S3_BUCKET/models/weights/rsgpt/finetune_50k.pth"

# Vocab data
if [[ -d "$RASYN_DIR/data/vocab" ]]; then
    echo "[bonus] Uploading vocab data..."
    aws s3 sync "$RASYN_DIR/data/vocab/" "s3://$S3_BUCKET/models/data/vocab/"
fi

echo ""
echo "=== Upload complete! ==="
echo ""
echo "S3 contents:"
aws s3 ls "s3://$S3_BUCKET/models/" --recursive --human-readable --summarize
