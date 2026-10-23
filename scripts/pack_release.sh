#!/usr/bin/env bash
# Build a self-contained tarball for offline transfer to a server
# (use when the server is not directly reachable via SSH from here).
#
#   ./scripts/pack_release.sh [direction]      # -> dist/tradingbot-release.tar.gz
#
# On the server: unpack, then run install.sh (prints the exact steps).
set -euo pipefail
DIRECTION="${1:-long}"
cd "$(dirname "$0")/.."
TAG="$(date +%Y%m%d-%H%M)"
OUT="dist/tradingbot-$DIRECTION-$TAG.tar.gz"
mkdir -p dist
tar -czf "$OUT" \
  --exclude=".venv" --exclude="__pycache__" --exclude=".idea" \
  --exclude=".git" --exclude="dataset/candles_live" --exclude="results/logs" \
  --exclude="results/live_state.json" --exclude=".env" \
  --exclude="dist" --exclude=".venv*venv" \
  config.py main.py main_live.py requirements-release.txt \
  docker-compose.yml .env.example \
  deploy scripts docs \
  src config results/fits/$DIRECTION dataset/candles_live dataset/candles \
  || true
echo "release: $OUT"
echo "contents check:"
tar -tzf "$OUT" | grep -E "results/fits|deploy/|docker-compose" | head
