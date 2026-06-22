FROM python:3.11-slim

WORKDIR /app

# System deps for lightgbm / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached unless pyproject changes)
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Volumes for data and artifacts (mount at runtime)
VOLUME ["/app/data", "/app/artifacts"]

# Default: show CLI help
ENTRYPOINT ["pump"]
CMD ["--help"]
