"""Live logging: append-only JSONL for signals and trades + console setup."""
import json
import logging
import os
import time
from pathlib import Path

LOG_DIR = Path("results/logs")


def set_log_dir(path):
    """Point every JSONL writer at an isolated directory (per mode/direction)."""
    global LOG_DIR
    LOG_DIR = Path(path)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def setup_console(level=logging.INFO):
    logging.basicConfig(level=level,
                        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")


def _append(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), **record}
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def log_signal(record: dict):
    """One fully self-contained explanation per rule trigger per bar."""
    _append(LOG_DIR / "signals.jsonl", record)


def log_trade(record: dict):
    """Orders, fills, state transitions, exits, errors."""
    _append(LOG_DIR / "trades.jsonl", record)


def log_cycle(record: dict):
    _append(LOG_DIR / "cycles.jsonl", record)


def read_signals(limit: int = 20) -> list:
    path = LOG_DIR / "signals.jsonl"
    if not path.exists():
        return []
    lines = open(path, "r", encoding="utf8").readlines()
    return [json.loads(l) for l in lines[-limit:]]
