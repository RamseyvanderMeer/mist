# Multi-stage build for MIST scraper - target size < 500MB
# Stage 1: Builder
FROM python:3.12-alpine AS builder

WORKDIR /build

# Install build deps for psycopg2
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    postgresql-dev \
    linux-headers

# Create venv and install deps
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements-scraper.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-scraper.txt

# Stage 2: Runtime
FROM python:3.12-alpine AS runtime

WORKDIR /app

# Runtime deps only (no gcc/musl-dev)
RUN apk add --no-cache \
    libpq \
    curl

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY scrapers/ ./scrapers/
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/run_scraper.py ./run_scraper.py

# Scrapy project config
COPY scrapy.cfg ./

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Cloud Run Jobs expect this
CMD ["python", "run_scraper.py"]
