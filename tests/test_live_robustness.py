import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from src.data.source import CANDLE_COLUMNS
from src.execution.wallex import (DryRunPortfolio, WallexClient,
                                  WallexClientError, _decimals_from_step)
from src.live.evaluator import default_warmup_bars
from src.live.trader import LiveTrader


def test_default_warmup_covers_30d_horizon():
    assert default_warmup_bars([3, 12, 48, 288, 2016, 8640]) == 17568


def test_step_decimals_are_parsed_defensively():
    assert _decimals_from_step(8) == 8
    assert _decimals_from_step("0.00001") == 5
    assert _decimals_from_step(None) == 6


def test_short_paper_accounting_allows_negative_holdings():
    p = DryRunPortfolio(1000.0)
    p.apply_fill("AAAUSDT", "SELL", 1.0, 100.0)
    assert p.holdings["AAAUSDT"] == -1.0
    assert p.equity({"AAAUSDT": 90.0}) > 1000.0   # short profits as price falls
    p.apply_fill("AAAUSDT", "BUY", 1.0, 90.0)
    assert abs(p.holdings["AAAUSDT"]) < 1e-12


class _Resp:
    def __init__(self, status=200, payload=None, ok=True, text=""):
        self.status_code = status
        self.ok = ok
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _Session:
    def __init__(self, failures, payload):
        self.failures = failures
        self.payload = payload
        self.calls = 0

    def request(self, *a, **k):
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise requests.ConnectionError("network down")
        return _Resp(payload=self.payload)

    def close(self):
        pass


def _markets_payload():
    return {"result": {"symbols": {"BTCUSDT": {
        "baseAsset": "BTC", "quoteAsset": "USDT", "stepSize": 8,
        "minQty": 1e-8, "minNotional": 1, "isMarketTypeEnable": True,
        "stats": {"lastPrice": "100", "24h_quoteVolume": "1000"}}}},
        "success": True}


def test_client_retries_then_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    c = WallexClient(max_retries=2, market_cache_path=str(tmp_path / "m.json"))
    monkeypatch.setattr(c, "_reset_session", lambda: None)
    c.session = _Session(failures=2, payload=_markets_payload())
    markets = c.get_markets(force=True)
    assert "BTCUSDT" in markets
    assert c.consecutive_failures == 0
    assert c.session.calls == 3


def test_client_uses_disk_markets_cache_while_offline(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    cache = tmp_path / "m.json"
    cache.write_text(json.dumps(_markets_payload()["result"]["symbols"]))
    c = WallexClient(max_retries=1, market_cache_path=str(cache))
    monkeypatch.setattr(c, "_reset_session", lambda: None)
    c.session = _Session(failures=99, payload=_markets_payload())
    markets = c.get_markets(force=True)   # once-fresh cache, forced refresh fails
    assert "BTCUSDT" in markets
    assert c.consecutive_failures > 0


def test_client_fails_fast_on_auth_error(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    c = WallexClient(api_key="x", max_retries=3,
                     market_cache_path=str(tmp_path / "m.json"))
    inst = _Session(failures=0, payload={})
    calls = {"n": 0}

    def req(*a, **k):
        calls["n"] += 1
        return _Resp(status=401, payload={}, ok=False, text="unauthorized")

    inst.request = req
    c.session = inst
    with pytest.raises(WallexClientError):
        c.request("GET", "/v1/markets")
    assert calls["n"] == 1


class _FakeSource:
    def __init__(self, frames):
        self.frames = frames

    def read_pair(self, pair, **kwargs):
        return self.frames[pair]


def _random_walk(n, seed, start=100.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.001, n)
    close = start * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0008, n)))
    open_ = np.r_[close[0], close[:-1]]
    dt = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({"datetime": dt, "open": open_, "high": high,
                         "low": low, "close": close,
                         "volume": rng.uniform(1, 10, n)})[CANDLE_COLUMNS]


def test_evaluator_has_horizon_and_btc_context_with_full_warmup():
    from src.live.evaluator import LiveEvaluator

    n = 18000
    src = _FakeSource({"BTCUSDT": _random_walk(n, 1, 50000),
                       "ETHUSDT": _random_walk(n, 2, 3000)})
    class DummyDisc:
        feature_info = []
        bins = {}
    ev = LiveEvaluator(DummyDisc(), src)
    assert ev.require_bars == 17568

    eth = ev.features_df("ETHUSDT")
    assert len(eth) == 17856
    last = eth.iloc[-1]
    for col in ("h_momz_30d", "h_cumret_7d", "btc_logret_12",
                "symbol_relative_return_288", "symbol_btc_corr_288"):
        assert col in eth.columns, col
        assert np.isfinite(last[col]), col

    btc = ev.features_df("BTCUSDT")
    last = btc.iloc[-1]
    assert last["symbol_relative_return_288"] == 0.0
    assert last["symbol_btc_corr_288"] == 1.0
    assert np.isfinite(last["btc_logret_12"])


def test_evaluator_replay_tail_requests_enough_history():
    from src.live.evaluator import LiveEvaluator

    n = 18000
    src = _FakeSource({"BTCUSDT": _random_walk(n, 1), "ETHUSDT": _random_walk(n, 2)})
    class DummyDisc:
        feature_info = []
        bins = {}
    ev = LiveEvaluator(DummyDisc(), src)
    with pytest.raises(ValueError, match="warm-up \\+ replay"):
        ev.features_df("ETHUSDT", tail_bars=20000)


def test_staleness_is_bar_based():
    cutoff = pd.Timestamp("2026-01-01 10:00")
    fresh = pd.DataFrame({"datetime": [pd.Timestamp("2026-01-01 09:55")]})
    stale = pd.DataFrame({"datetime": [pd.Timestamp("2026-01-01 09:30")]})
    assert LiveTrader._is_stale(None, fresh, cutoff) is False
    assert LiveTrader._is_stale(None, stale, cutoff) is True
