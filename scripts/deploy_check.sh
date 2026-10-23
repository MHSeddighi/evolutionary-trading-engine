#!/usr/bin/env bash
# Pre-deployment safety gate: everything must pass before real money.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/6 compile check =="
.venv/bin/python -m py_compile main_live.py src/execution/wallex.py \
  src/data/wallex_source.py src/data/database_source.py src/data/live_store.py \
  src/live/fit_package.py src/live/logger.py src/live/state.py \
  src/live/evaluator.py src/live/trader.py

echo "== 2/6 candle DB up (docker) =="
docker compose up -d candles-db

echo "== 3/6 fit package present =="
test -s "results/fits/${1:-long}/fit_package.pkl" || {
  echo "MISSING fit package — run: .venv/bin/python scripts/package_fit.py --direction ${1:-long}"; exit 1; }

echo "== 4/6 ruleset present =="
test -s "results/fits/${1:-long}/ruleset.json" || {
  echo "MISSING ruleset — write results/fits/${1:-long}/ruleset.json (schema in docs/PHASE2.md)"; exit 1; }

echo "== 5/6 single paper cycle =="
.venv/bin/python main_live.py --dry-run --once --symbols auto --max-symbols 12

echo "== 6/6 logs written =="
test -s results/logs/cycles.jsonl
echo "ALL CHECKS PASSED — safe to systemctl enable --now tradingbot (paper), then --live when ready."
