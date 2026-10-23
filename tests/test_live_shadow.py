import json
from pathlib import Path

import pandas as pd
import pytest

from src.live.shadow import ShadowBroker


def _bars(rows, sym="AAAUSDT"):
    df = pd.DataFrame(rows)
    df["datetime"] = pd.date_range("2026-01-01", periods=len(df), freq="5min")
    df["volume"] = 1.0
    return {sym: df}


def _trigger(sym="AAAUSDT", direction="long", cap=100.0, tp=2.0, sl=1.0):
    return {"symbol": sym, "direction": direction, "rule_id": 1,
            "name": "t", "conditions": ["[x] IS High"], "condition_detail": [],
            "tp": tp, "sl": sl, "capital_pct": cap}


def test_long_take_profit_and_stop_loss(tmp_path):
    b = ShadowBroker(initial_usdt=1000.0, log_dir=tmp_path)
    pos = b.submit(_trigger(), 100.0, "2026-01-01 00:00")
    assert pos and pos["entry_fill"] > 100.0      # fee + adverse slip on buy
    rec = b._check_exit(pos, pd.DataFrame([{
        "datetime": pd.Timestamp("2026-01-01 00:05"), "open": 100.0,
        "high": 103.0, "low": 99.5, "close": 102.0, "volume": 1}]))
    assert rec["exit_reason"] == "take_profit"
    assert rec["pnl"] > 0 and b.equity > 1000.0

    b2 = ShadowBroker(initial_usdt=1000.0, log_dir=tmp_path)
    p2 = b2.submit(_trigger(), 100.0, "2026-01-01 00:00")
    rec2 = b2._check_exit(p2, pd.DataFrame([{
        "datetime": pd.Timestamp("2026-01-01 00:05"), "open": 100.0,
        "high": 100.2, "low": 98.0, "close": 98.5, "volume": 1}]))
    assert rec2["exit_reason"] == "stop_loss"
    assert rec2["pnl"] < 0 and b2.equity < 1000.0


def test_short_sides_are_mirrored(tmp_path):
    b = ShadowBroker(initial_usdt=1000.0, log_dir=tmp_path)
    pos = b.submit(_trigger(direction="short", tp=2.0, sl=2.0), 100.0,
                   "2026-01-01 00:00")
    assert pos["entry_fill"] < 100.0              # short sells at a worse price
    up = b._check_exit(pos, pd.DataFrame([{
        "datetime": pd.Timestamp("2026-01-01 00:05"), "open": 100.0,
        "high": 103.0, "low": 99.9, "close": 102.0, "volume": 1}]))
    assert up["exit_reason"] == "stop_loss" and up["pnl"] < 0

    b2 = ShadowBroker(initial_usdt=1000.0, log_dir=tmp_path)
    p2 = b2.submit(_trigger(direction="short", tp=2.0, sl=2.0), 100.0,
                   "2026-01-01 00:00")
    down = b2._check_exit(p2, pd.DataFrame([{
        "datetime": pd.Timestamp("2026-01-01 00:05"), "open": 100.0,
        "high": 100.1, "low": 97.0, "close": 98.0, "volume": 1}]))
    assert down["exit_reason"] == "take_profit" and down["pnl"] > 0


def test_gap_through_stop_fills_at_open(tmp_path):
    b = ShadowBroker(initial_usdt=1000.0, log_dir=tmp_path)
    pos = b.submit(_trigger(sl=1.0), 100.0, "2026-01-01 00:00")
    rec = b._check_exit(pos, pd.DataFrame([{
        "datetime": pd.Timestamp("2026-01-01 00:05"), "open": 95.0,
        "high": 96.0, "low": 94.0, "close": 95.0, "volume": 1}]))
    assert rec["exit_reason"] == "stop_loss"
    assert rec["exit_price"] < 96.0               # filled at the gap open, not the stop


def test_capital_is_reserved_across_positions(tmp_path):
    b = ShadowBroker(initial_usdt=1000.0, log_dir=tmp_path)
    assert b.submit(_trigger(sym="AAAUSDT", cap=80), 100.0, "t0")
    assert b.submit(_trigger(sym="BBBUSDT", cap=80), 100.0, "t0")
    assert b._open_alloc() <= 1000.0 + 1e-9
    assert not b.submit(_trigger(sym="CCCUSDT", cap=80), 100.0, "t0")


def test_outage_replays_missed_bars(tmp_path):
    b = ShadowBroker(initial_usdt=1000.0, log_dir=tmp_path)
    b.submit(_trigger(sl=1.0), 100.0, "2026-01-01 00:00")
    bars = _bars([
        {"open": 100, "high": 100.1, "low": 99.9, "close": 100},
        {"open": 100, "high": 100.0, "low": 97.0, "close": 99},   # stop here
        {"open": 99, "high": 105.0, "low": 99.0, "close": 104},    # recovery
    ])
    closed = b.on_bars(bars)
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "stop_loss"


def test_end_of_data_close_and_stats(tmp_path):
    b = ShadowBroker(initial_usdt=1000.0, log_dir=tmp_path)
    b.submit(_trigger(tp=50), 100.0, "2026-01-01 00:00")
    closed = b.close_open({"AAAUSDT": 101.0})
    assert closed and closed[0]["exit_reason"] == "end_of_data"
    stats = b.snapshot()
    assert stats["closed_trades"] == 1
    saved = json.loads((Path(tmp_path) / "shadow_stats.json").read_text())
    assert saved["direction"] == "long"
