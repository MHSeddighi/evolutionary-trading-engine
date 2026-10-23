#!/usr/bin/env bash
# Server-side launcher used by the systemd unit.
# honors two env vars (set in /etc/tradingbot.env):
#   MODE=live|paper (.paper = --dry-run), DIRECTION=long|short
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${MODE:-paper}"
DIRECTION="${DIRECTION:-long}"
ARGS=(--direction "$DIRECTION" --symbols auto --max-symbols 12)
if [[ "$MODE" == "live" ]]; then
  : "${WALLEX_API_KEY:?WALLEX_API_KEY must be set in /etc/tradingbot.env}"
  ARGS+=(--live)
else
  ARGS+=(--dry-run)
fi
exec .venv/bin/python main_live.py "${ARGS[@]}"
