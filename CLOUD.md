# Cloud & Headless Training Guide

Run PokeRL training without the GUI — locally, on a cloud VM, or in Docker.

## Option 1: Launch Scripts (Recommended)

The simplest approach. No Docker needed — just Python, Node.js, and git.

### Windows

```
start_training.bat
```

### Linux / macOS / Cloud VM

```bash
./start_training.sh
```

These scripts automatically:
- Clone and build Pokemon Showdown (first run only)
- Start the Showdown server
- Launch training with auto-resume from the latest checkpoint
- Shut down cleanly on Ctrl+C

### Passing Extra Arguments

```bash
# Windows
start_training.bat --infinite --total-battles 50000

# Linux/macOS
./start_training.sh --infinite --total-battles 50000
```

### Monitoring

```bash
# Follow the training log (Linux/macOS)
tail -f logs/training.log

# Windows (PowerShell)
Get-Content logs\training.log -Wait
```

### Stopping and Resuming

Press **Ctrl+C** to stop. Just run the script again to resume — it automatically
loads the latest checkpoint.

---

## Option 2: Docker

If you have Docker available, you can run everything in a container.

### Quick Start

```bash
# Build and start training
docker compose up --build

# Run in the background
docker compose up --build -d
```

Training automatically resumes from the latest checkpoint on every start.

### Passing Extra Arguments

```bash
docker compose run pokerl --infinite
docker compose run pokerl --total-battles 50000
docker compose run pokerl --device cuda
```

### Monitoring

```bash
tail -f logs/training.log
docker compose ps
docker compose logs -f
```

### File Layout

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./checkpoints/` | `/app/checkpoints/` | Saved model checkpoints |
| `./logs/` | `/app/logs/` | Training log files |
| `./teams/` | `/app/teams/` | Team files (editable without rebuild) |

### Stopping and Resuming

```bash
docker compose down       # stop
docker compose up -d      # resume
```

The container is configured with `restart: unless-stopped`, so it auto-restarts
after crashes or Docker daemon restarts.

---

## Deploying on a Cloud VM

### 1. Launch a VM

Any cloud provider works (AWS EC2, GCP Compute Engine, DigitalOcean, etc.):
- **CPU training**: 2+ vCPUs, 4+ GB RAM (e.g., AWS `t3.medium`, ~$30/month)
- **GPU training**: Any GPU instance with NVIDIA drivers

### 2. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip nodejs npm git

# Clone repo
git clone <your-repo-url> PokeRL
cd PokeRL
pip install -r requirements.txt
```

### 3. Start Training in Background

Use `screen` or `tmux` so training survives SSH disconnects:

```bash
# Start a persistent screen session
screen -S pokerl

# Launch training
./start_training.sh --infinite

# Detach: press Ctrl+A, then D
# Reattach later:
screen -r pokerl
```

### 4. Download Checkpoints

```bash
# From your local machine
scp -r user@vm-ip:~/PokeRL/checkpoints/ ./checkpoints/
```

---

## GPU Support

### Without Docker

```bash
# Just pass --device cuda
./start_training.sh --device cuda
```

Requires PyTorch with CUDA support (`pip install torch --index-url https://download.pytorch.org/whl/cu121`).

### With Docker

Add NVIDIA runtime to `docker-compose.yml`:

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

Then: `docker compose run pokerl --device cuda`

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHOWDOWN_PORT` | `8000` | Pokemon Showdown server port |
| `LOG_FILE` | `logs/training.log` | Log file path |
