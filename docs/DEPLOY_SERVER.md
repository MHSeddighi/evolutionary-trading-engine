# Server deployment (24/7)

Two paths, both already scripted:

## A. Server reachable via SSH from this machine

```bash
# fresh paper deployment
./scripts/deploy_server.sh user@server --mode paper

# later, live trading
export WALLEX_API_KEY=...
./scripts/deploy_server.sh user@server --mode live
```

Idempotent: re-run any time to push updated code/rules/fits. It does:
rsync → venv → candle DB (docker) + CSV migration → smoke cycle → systemd
service (`/etc/systemd/system/tradingbot.service`, restarts on crash,
starts on boot). Mode lives in `/etc/tradingbot.env` (`MODE=paper|live`,
`WALLEX_API_KEY`, chmod 600).

Day-2 operations on the server:

```bash
journalctl -u tradingbot -f                        # service logs
tail -f /opt/trading-bot/results/logs/signals.jsonl  # explained signals
tail -f /opt/trading-bot/results/logs/cycles.jsonl   # latency + paper equity
sudo systemctl restart tradingbot                  # restart / mode switch
```

## B. No direct SSH (offline transfer)

```bash
./scripts/pack_release.sh long                 # -> dist/tradingbot-long-*.tar.gz
scp dist/tradingbot-long-*.tar.gz user@server:
# on the server:
tar -xzf tradingbot-long-*.tar.gz -C ~tradingbot-install
cd ~tradingbot-install && sudo bash deploy/install.sh long
```

`install.sh` = venv + docker DB + migration + deploy_check + systemd.

## Critical maintenance rule

Phase 1 (the strategy) and its registry are still evolving in parallel. After
any phase-1 change, on this machine:

```bash
.venv/bin/python scripts/package_fit.py --direction long   # re-export bins
.venv/bin/python scripts/build_ruleset.py --direction long # re-pick strategies
./scripts/deploy_server.sh user@server --mode paper        # or pack_release.sh
```

Stale bins = live "Very High" bands meaning a different quantile than in
research — the runbook above prevents that. The live evaluator skips any rule
whose features it cannot compute (logged, never crashes).
