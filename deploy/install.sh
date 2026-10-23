#!/usr/bin/env bash
# Offline install of a tradingbot release tarball on the target server.
# Usage:  sudo bash install.sh [direction]   (run from the unpacked directory)
set -euo pipefail
DIRECTION="${1:-long}"
DEST=/opt/trading-bot
SELF="$(cd "$(dirname "$0")" && pwd)"

command -v docker >/dev/null || { echo "docker required"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }

echo "==> [1/5] install to $DEST (rsync-style refresh, keeps runtime state)"
mkdir -p "$DEST"
rsync -a --exclude=results/live_state.json --exclude=results/logs \
      --exclude=dataset/candles_live "$SELF/" "$DEST/"

echo "==> [2/5] venv + deps"
cd "$DEST"
test -d .venv || python3 -m venv .venv
.venv/bin/pip -q install --upgrade pip
.venv/bin/pip -q install -r requirements-release.txt

echo "==> [3/5] candle DB + migration"
docker compose up -d candles-db
sleep 5
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, '.')
from src.data.database_source import DatabaseSource
print('migrated:', DatabaseSource().migrate_csvs())
EOF

echo "==> [4/5] pre-flight checks"
bash scripts/deploy_check.sh "$DIRECTION"

echo "==> [5/5] systemd"
RUN_USER="${SUDO_USER:-$(whoami)}"
sed "s/User=%i/User=$RUN_USER/" deploy/tradingbot.service > /tmp/tradingbot.service
mv /tmp/tradingbot.service /etc/systemd/system/tradingbot.service
MODE=${MODE:-paper}; DIRECTION=${DIRECTION:-$DIRECTION}
printf 'MODE=%s\nDIRECTION=%s\n' "$MODE" "$DIRECTION" > /etc/tradingbot.env
chmod 600 /etc/tradingbot.env
systemctl daemon-reload
systemctl enable --now tradingbot
systemctl is-active tradingbot
echo
echo "installed (mode=$MODE). logs:  journalctl -u tradingbot -f"
echo "go live later:  echo 'WALLEX_API_KEY=...' >> /etc/tradingbot.env; "
echo "                sed -i 's/MODE=paper/MODE=live/' /etc/tradingbot.env; systemctl restart tradingbot"
