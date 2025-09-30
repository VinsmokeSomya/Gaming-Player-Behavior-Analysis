# Multi-stage build for production deployment
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r dashuser && useradd -r -g dashuser dashuser

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/dashuser/.local

# Copy application code
COPY . .

# Set ownership
RUN chown -R dashuser:dashuser /app

# Switch to non-root user
USER dashuser

# Set environment variables
ENV PATH=/home/dashuser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV DASH_HOST=0.0.0.0
ENV DASH_PORT=8050
ENV DASH_DEBUG=False

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8050/dashboard/ || exit 1

# Expose port
EXPOSE 8050

# Run the application
CMD ["python", "app.py"]