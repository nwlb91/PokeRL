FROM python:3.11-slim

# Install Node.js and git (needed for Pokemon Showdown)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Clone and build Pokemon Showdown
RUN git clone --depth 1 https://github.com/smogon/pokemon-showdown.git /app/pokemon-showdown && \
    cd /app/pokemon-showdown && \
    npm install && \
    node build

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

RUN chmod +x docker-entrypoint.sh

VOLUME ["/app/checkpoints", "/app/logs"]

ENTRYPOINT ["./docker-entrypoint.sh"]
