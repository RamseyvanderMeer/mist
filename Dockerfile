# Dockerfile for MIST API
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
# Build: Mon Mar 30 06:08:13 PM PDT 2026
# Build: Mon Mar 30 06:56:26 PM PDT 2026
# Build: Mon Mar 30 07:17:39 PM PDT 2026
# Build: Tue Mar 31 12:16:31 AM PDT 2026
# Build: Tue Mar 31 12:40:00 AM PDT 2026
# Build: Tue Mar 31 12:58:15 AM PDT 2026
# Build: Tue Mar 31 01:34:41 AM PDT 2026
