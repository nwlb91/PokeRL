# Cloud Training Guide

Run PokeRL training in a Docker container so it keeps going even when your local machine is off.

## Quick Start (Local Docker)

```bash
# Build and start training
docker compose up --build

# Run in the background
docker compose up --build -d
```

Training automatically resumes from the latest checkpoint on every start.

## Passing Extra Arguments

Any arguments after the image name are forwarded to `train.py`:

```bash
# Train forever
docker compose run pokerl --infinite

# Train for a specific number of battles
docker compose run pokerl --total-battles 50000

# Use GPU
docker compose run pokerl --device cuda
```

## Monitoring

```bash
# Follow the training log
tail -f logs/training.log

# Check container status
docker compose ps

# View live container output
docker compose logs -f
```

## File Layout

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./checkpoints/` | `/app/checkpoints/` | Saved model checkpoints |
| `./logs/` | `/app/logs/` | Training log files |
| `./teams/` | `/app/teams/` | Team files (editable without rebuild) |

## Stopping and Resuming

```bash
# Graceful stop
docker compose down

# Resume (automatically loads latest checkpoint)
docker compose up -d
```

The container is configured with `restart: unless-stopped`, so it will automatically restart after crashes or Docker daemon restarts.

## Deploying on a Cloud VM

### 1. Launch a VM

Any cloud provider works (AWS EC2, GCP Compute Engine, DigitalOcean, etc.). Recommended:
- **CPU training**: 2+ vCPUs, 4+ GB RAM (e.g., AWS `t3.medium`)
- **GPU training**: Any GPU instance with NVIDIA drivers + Docker

### 2. Install Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in, then:
docker compose version  # verify install
```

### 3. Clone and Start

```bash
git clone <your-repo-url> PokeRL
cd PokeRL

# Edit teams if needed
# vim teams/team1.txt
# vim teams/team2.txt

# Build and run in background
docker compose up --build -d

# Check logs
tail -f logs/training.log
```

### 4. Download Checkpoints

```bash
# From your local machine
scp -r user@vm-ip:~/PokeRL/checkpoints/ ./checkpoints/
```

## GPU Support

For GPU training, modify `docker-compose.yml` to add the NVIDIA runtime:

```yaml
services:
  pokerl:
    build: .
    container_name: pokerl-training
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./teams:/app/teams
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Then run with `--device cuda`:

```bash
docker compose run pokerl --device cuda
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHOWDOWN_PORT` | `8000` | Pokemon Showdown server port (internal) |
| `LOG_FILE` | `/app/logs/training.log` | Log file path |
