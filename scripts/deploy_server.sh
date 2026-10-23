#!/usr/bin/env bash
# Deploy the trading bot to a remote server for 24/7 operation.
#
# Usage:
#   export WALLEX_API_KEY=...          # only needed for --mode live
#   ./scripts/deploy_server.sh user@server [--mode paper|live] [--direction long]
#
# Idempotent; can be re-run to push updated fits/rules/code:
#   1. rsync repo -> /opt/trading-bot (results/fits/* included: fit pkg + ruleset)
#   2. python venv + deps on the server
#   3. dockerized candle DB up + CSV cache migration
#   4. single paper smoke cycle
#   5. systemd unit installed, service enabled & restarted
#   6. mode (paper/live) comes from /etc/tradingbot.env on the server
set -euo pipefail

SERVER="${1:?usage: deploy_server.sh user@server [--mode paper|live] [--direction long]}"
shift || true
MODE=paper
DIRECTION=long
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --direction) DIRECTION="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

DEST=/opt/trading-bot
RUN_USER="$(echo "$SERVER" | cut -d@ -f1)"

if [[ "$MODE" == "live" ]]; then
  [[ -n "${WALLEX_API_KEY:-}" ]] || { echo "ERROR: export WALLEX_API_KEY for --mode live"; exit 1; }
fi

echo "==> [1/6] rsync repo -> $SERVER:$DEST"
rsync -a \
  --exclude=.venv --exclude=__pycache__ --exclude=.idea \
  --exclude=dataset/candles_live --exclude=results/logs \
  --exclude=results/live_state.json --exclude=.env --exclude=\.git \
  ./ "$SERVER:$DEST/"
ssh "$SERVER" "chmod +x $DEST/scripts/*.sh $DEST/deploy/start_trader.sh"

echo "==> [2/6] python venv + deps"
ssh "$SERVER" "cd $DEST && bash -s" <<'REMOTE'
set -e
command -v python3 >/dev/null || { echo "python3 missing"; exit 1; }
python3 -m venv .venv 2>/dev/null || true
PY=.venv/bin/python
$PY - <<'EOF' && exit 0 || true
import importlib.util
need = ["pandas", "numpy", "numba", "psycopg", "pyarrow", "sklearn", "requests", "pymoo", "xgboost"]
sys.exit(0 if all(importlib.util.find_spec(m) for m in need) else 1)
EOF
.venv/bin/pip -q install --upgrade pip
.venv/bin/pip -q install 'psycopg[binary]' pandas numpy numba scikit-learn pyarrow xgboost pymoo requests
REMOTE

echo "==> [3/6] candle DB up + migration"
ssh "$SERVER" "cd $DEST && docker compose up -d candles-db && sleep 5 && \
  .venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, '.')
from src.data.database_source import DatabaseSource
print('migrated:', DatabaseSource().migrate_csvs())
EOF"

echo "==> [4/6] smoke test (single paper cycle, keyless)"
ssh "$SERVER" "cd $DEST && \
  test -s results/fits/$DIRECTION/fit_package.pkl && \
  test -s results/fits/$DIRECTION/ruleset.json && \
  .venv/bin/python main_live.py --dry-run --once --symbols BTCUSDT --max-symbols 1"

echo "==> [5/6] mode configuration (/etc/tradingbot.env)"
if [[ "$MODE" == "live" ]]; then
  ssh "$SERVER" "printf 'MODE=%s\nDIRECTION=%s\nWALLEX_API_KEY=%s\n' \
      '$MODE' '$DIRECTION' '$WALLEX_API_KEY' | \
      sudo tee /etc/tradingbot.env >/dev/null && sudo chmod 600 /etc/tradingbot.env"
else
  ssh "$SERVER" "printf 'MODE=%s\nDIRECTION=%s\n' '$MODE' '$DIRECTION' | \
      sudo tee /etc/tradingbot.env >/dev/null"
fi

echo "==> [6/6] systemd install + start"
sed "s/User=%i/User=$RUN_USER/" deploy/tradingbot.service > /tmp/tradingbot.service
scp -q /tmp/tradingbot.service "$SERVER:/tmp/tradingbot.service"
ssh "$SERVER" "sudo mv /tmp/tradingbot.service /etc/systemd/system/tradingbot.service && \
  sudo systemctl daemon-reload && sudo systemctl enable tradingbot && \
  sudo systemctl restart tradingbot && sleep 3 && \
  echo 'service state:' && systemctl is-active tradingbot"

echo
echo "Deployed to $SERVER (mode=$MODE, direction=$DIRECTION)"
echo "  logs:      ssh $SERVER journalctl -u tradingbot -f"
echo "  signals:   ssh $SERVER tail -f $DEST/results/logs/signals.jsonl"
echo "  status:    ssh $SERVER systemctl status tradingbot"
echo "  switch mode later: edit /etc/tradingbot.env (MODE=live + WALLEX_API_KEY) then sudo systemctl restart tradingbot"
