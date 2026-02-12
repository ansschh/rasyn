FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies for RDKit SVG rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 curl && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code only (checkpoints/weights excluded via .dockerignore)
COPY rasyn/ ./rasyn/
COPY configs/ ./configs/
COPY setup.py pyproject.toml README.md ./
RUN pip install --no-cache-dir -e . 2>/dev/null || true

# Health check — ALB also checks this, but Docker health is useful for debugging
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "rasyn.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
