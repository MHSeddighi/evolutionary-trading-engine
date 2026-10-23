"""Phase 2 execution: Wallex REST client + executor (live or dry-run).

Public endpoints (no key):  GET /v1/markets, GET /v1/udf/history, GET /v1/depth,
                            GET /v1/trades
Private endpoints (X-API-Key header): account balances, orders, openOrders.

Dry-run mode (default) never sends private requests: fills are simulated at the
current last/bid price and tracked in a local paper portfolio so the whole
signal -> trade chain runs without risking money.

Rate limit: Wallex allows ~100 req/s; this client paces at max_rate/s with
retries + exponential backoff on 429/5xx.
"""
from abc import ABC, abstractmethod
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.wallex.ir"
RESOLUTION_SECONDS = {"1": 60, "15": 900, "60": 3600, "240": 14400,
                      "720": 43200, "1D": 86400}
MARKET_CACHE_PATH = Path("results/cache/markets.json")
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}


class WallexClientError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class WallexClient:
    """Thin, defensive REST wrapper around api.wallex.ir.

    Outage behaviour:
      - 408/425/429/5xx and connection errors retry with exponential backoff
        + jitter; 4xx auth/validation errors fail fast.
      - Market metadata is cached on disk. If the network is down at startup
        (or later), the stale cache keeps symbol/precision logic alive so the
        trader can enter a degraded "no new entries" state instead of crashing.
      - Last known prices are cached in memory for paper equity marking.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: float = 15.0,
                 max_rate: float = 20.0, max_retries: int = 4,
                 markets_ttl: float = 300.0,
                 market_cache_path: str = str(MARKET_CACHE_PATH)):
        self.api_key = api_key
        self.timeout = timeout
        self.max_rate = max_rate
        self.max_retries = max_retries
        self._min_interval = 1.0 / max_rate
        self._last_ts = 0.0
        self._lock = threading.Lock()
        self.session = requests.Session()
        self._markets: Optional[Dict] = None
        self._markets_ts = 0.0
        self.markets_ttl = markets_ttl
        self.market_cache_path = Path(market_cache_path)
        self._last_prices: Dict[str, tuple] = {}   # symbol -> (price, ts)
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0
        self._load_market_cache()

    # ---------------- health / cache ----------------

    @property
    def healthy(self) -> bool:
        return self.consecutive_failures == 0

    def _reset_session(self):
        try:
            self.session.close()
        except Exception:
            pass
        self.session = requests.Session()

    def _load_market_cache(self):
        try:
            if self.market_cache_path.exists():
                self._markets = json.loads(self.market_cache_path.read_text())
                self._markets_ts = self.market_cache_path.stat().st_mtime
                logger.info("Loaded cached market metadata (%d symbols) from %s",
                            len(self._markets), self.market_cache_path)
        except Exception as e:
            logger.warning("Could not read market cache: %s", e)

    def _save_market_cache(self):
        try:
            self.market_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.market_cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._markets))
            tmp.replace(self.market_cache_path)
        except Exception as e:
            logger.warning("Could not persist market cache: %s", e)

    # ---------------- core request ----------------

    def request(self, method: str, path: str, params: Dict = None,
                body: Dict = None, auth: bool = False) -> dict:
        url = BASE_URL + path
        headers = {"Content-Type": "application/json"}
        if auth:
            if not self.api_key:
                raise WallexClientError(
                    "Private endpoint requested without WALLEX_API_KEY "
                    "(refusing: set the env var or use --dry-run)")
            headers["X-API-Key"] = self.api_key
        last_exc = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            with self._lock:
                wait = self._last_ts + self._min_interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last_ts = time.monotonic()
            try:
                r = self.session.request(method, url, params=params,
                                         json=body, headers=headers,
                                         timeout=self.timeout)
                if r.status_code in RETRYABLE_HTTP:
                    raise WallexClientError(
                        f"HTTP {r.status_code}: {r.text[:200]}", retryable=True)
                data = r.json()
                if isinstance(data, dict) and data.get("success") is False:
                    code = (data.get("result") or {}).get("error_code")
                    raise WallexClientError(
                        f"Wallex error {code}: {data.get('message')}",
                        retryable=False)
                if not r.ok:
                    raise WallexClientError(
                        f"HTTP {r.status_code}: {r.text[:200]}", retryable=False)
                self.consecutive_failures = 0
                self.last_error = None
                return data
            except requests.RequestException as e:
                last_exc = WallexClientError(str(e), retryable=True)
            except WallexClientError as e:
                last_exc = e
            except Exception as e:  # malformed JSON etc.
                last_exc = WallexClientError(f"{type(e).__name__}: {e}",
                                             retryable=True)
            self.consecutive_failures += 1
            self.last_error = str(last_exc)
            if not getattr(last_exc, "retryable", True) or attempt >= attempts - 1:
                raise last_exc
            sleep = min(30.0, (2.0 ** attempt) * 0.75) * (0.75 + random.random() / 2)
            logger.warning("Wallex %s %s failed (attempt %d/%d): %s — retry in %.1fs",
                           method, path, attempt + 1, attempts, last_exc, sleep)
            if attempt:
                self._reset_session()
            time.sleep(sleep)
        raise last_exc

    # ---------------- public ----------------

    def get_markets(self, force: bool = False) -> Dict[str, dict]:
        """{SYMBOL: {baseAsset, quoteAsset, stepSize, tickSize, minQty,
                     minNotional, isMarketTypeEnable, stats{24h_quoteVolume,...}}}

        Returns cached metadata during an outage. Raises only if no cache has
        ever been written and the network is unavailable.
        """
        fresh = self._markets and (time.time() - self._markets_ts) < self.markets_ttl
        if fresh and not force:
            return self._markets
        try:
            data = self.request("GET", "/v1/markets")
            symbols = ((data.get("result") or {}).get("symbols")) or \
                      (data.get("result") or {})
            markets = {k.upper(): v for k, v in symbols.items()
                       if isinstance(v, dict) and "baseAsset" in v}
            if markets:
                self._markets = markets
                self._markets_ts = time.time()
                self._save_market_cache()
            return self._markets or {}
        except WallexClientError as e:
            if self._markets:
                logger.error("Market refresh failed (%s) — using cached metadata "
                             "(%d symbols). Trading should stay in degraded mode "
                             "until connectivity returns.", e, len(self._markets))
                return self._markets
            raise

    def market_info(self, symbol: str) -> Optional[dict]:
        return self.get_markets().get(symbol.upper())

    def usdt_markets_by_volume(self, min_notional_vol: float = 0.0) -> List[str]:
        """USDT-quoted symbols sorted by 24h quote volume (desc)."""
        out = []
        for sym, info in self.get_markets().items():
            if not sym.endswith("USDT") or not info.get("isMarketTypeEnable", True):
                continue
            stats = info.get("stats") or {}
            vol = float(stats.get("24h_quoteVolume") or 0)
            if vol >= min_notional_vol:
                out.append((vol, sym))
        return [s for _, s in sorted(out, reverse=True)]

    def get_klines(self, symbol: str, resolution: str, start_ts: int,
                   end_ts: int) -> Dict[str, list]:
        """OHLCV arrays via UDF history: {t,o,h,l,c,v} (unix sec, candle start)."""
        data = self.request("GET", "/v1/udf/history", params={
            "symbol": symbol, "resolution": resolution,
            "from": int(start_ts), "to": int(end_ts)})
        res = data.get("result") if isinstance(data.get("result"), dict) else data
        if res.get("s") != "ok" or not res.get("t"):
            return {"t": [], "o": [], "h": [], "l": [], "c": [], "v": []}
        return {k: res.get(k, []) for k in ("t", "o", "h", "l", "c", "v")}

    def depth(self, symbol: str) -> dict:
        data = self.request("GET", "/v1/depth", params={"symbol": symbol})
        return data.get("result") or {}

    def last_price(self, symbol: str, allow_stale: bool = True) -> float:
        info = self.market_info(symbol)
        if info:
            stats = info.get("stats") or {}
            for k in ("lastPrice", "bidPrice", "askPrice"):
                px = stats.get(k)
                if px:
                    price = float(px)
                    self._last_prices[symbol] = (price, time.time())
                    return price
        try:
            depth = self.depth(symbol)
            ask = (depth.get("ask") or [{}])[0].get("price")
            bid = (depth.get("bid") or [{}])[0].get("price")
            px = ask or bid
            if px:
                price = float(px)
                self._last_prices[symbol] = (price, time.time())
                return price
        except WallexClientError:
            pass
        if allow_stale and symbol in self._last_prices:
            px, ts = self._last_prices[symbol]
            logger.warning("No live price for %s — using last known %.8f (%.0fs old)",
                           symbol, px, time.time() - ts)
            return px
        raise WallexClientError(f"No price available for {symbol}")

    # ---------------- private ----------------

    def get_balances(self) -> Dict[str, float]:
        """{ASSET: total_value} parsed defensively."""
        data = self.request("GET", "/v1/account/balances", auth=True)
        result = data.get("result") or {}
        out = {}
        items = result.get("balances") or result.get("assets") or result.get("list") or []
        for b in items:
            asset = (b.get("asset") or b.get("symbol") or "").upper()
            amount = b.get("value", b.get("balance", b.get("total")))
            if asset and amount is not None:
                out[asset] = float(amount)
        return out

    def place_order(self, symbol: str, side: str, quantity: float,
                    price: float = None, order_type: str = "MARKET",
                    client_id: str = None) -> dict:
        body = {"symbol": symbol, "type": order_type, "side": side,
                "quantity": quantity}
        if price is not None:
            body["price"] = price
        if client_id:
            body["client_id"] = client_id
        data = self.request("POST", "/v1/account/orders", body=body, auth=True)
        return data.get("result") or {}

    def get_order(self, client_order_id: str) -> dict:
        data = self.request("GET", f"/v1/account/orders/{client_order_id}", auth=True)
        return data.get("result") or {}

    def cancel_order(self, client_order_id: str) -> dict:
        data = self.request("DELETE", "/v1/account/orders",
                            params={"clientOrderId": client_order_id}, auth=True)
        return data.get("result") or {}

    def get_open_orders(self, symbol: str = None) -> list:
        params = {"symbol": symbol} if symbol else None
        data = self.request("GET", "/v1/account/openOrders", params=params, auth=True)
        return (data.get("result") or {}).get("orders") or []


class BaseExecutor(ABC):

    @abstractmethod
    def place_order(self, symbol: str, side: str, quantity: float, price: float = None) -> dict:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_balance(self) -> dict:
        raise NotImplementedError


class DryRunPortfolio:
    """Paper USDT portfolio for dry-run fills. Fees at 0.2% (Wallex flat fee).

    Negative holdings represent shorts (paper mode only): the SELL proceeds are
    credited to USDT and equity is marked at `usdt + holdings * price`.
    """

    FEE = 0.002

    def __init__(self, initial_usdt: float = 1000.0, state_path: str = None):
        self.initial_usdt = initial_usdt
        self.usdt = initial_usdt
        self.holdings: Dict[str, float] = {}
        self.realized_pnl = 0.0
        self.state_path = state_path

    def equity(self, prices: Dict[str, float]) -> float:
        return self.usdt + sum(q * prices.get(a, 0.0)
                               for a, q in self.holdings.items())

    def apply_fill(self, symbol: str, side: str, quantity: float, price: float):
        base = symbol  # track under the full symbol, e.g. BTCUSDT
        if side == "BUY":
            cost = quantity * price
            fee = cost * self.FEE
            self.usdt -= (cost + fee)
            self.realized_pnl -= fee
            self.holdings[base] = self.holdings.get(base, 0.0) + quantity
        else:
            proceeds = quantity * price
            fee = proceeds * self.FEE
            self.usdt += (proceeds - fee)
            self.realized_pnl -= fee
            # negative holdings = short; do not clamp to zero
            self.holdings[base] = self.holdings.get(base, 0.0) - quantity


def _decimals_from_step(step) -> int:
    """Wallex stepSize is a decimal count (8) but tolerate a tick string."""
    if step is None:
        return 6
    if isinstance(step, (int, float)):
        return max(0, int(step))
    s = str(step).strip()
    if "." in s:
        return max(0, len(s.split(".")[1]))
    try:
        return max(0, int(float(s)))
    except ValueError:
        return 6


class WallexExecutor(BaseExecutor):
    """Places real orders on Wallex, or simulates them when dry_run=True."""

    def __init__(self, client: WallexClient, dry_run: bool = True,
                 paper_usdt: float = 1000.0, markets: Optional[Dict] = None):
        self.client = client
        self.dry_run = dry_run
        self.paper = DryRunPortfolio(paper_usdt)
        self.markets = markets if markets is not None else client.get_markets()

    def refresh_markets(self):
        try:
            self.markets = self.client.get_markets(force=True)
        except Exception as e:
            logger.warning("Market metadata refresh failed: %s", e)

    def quantity_precision(self, symbol: str) -> int:
        info = self.markets.get(symbol) or {}
        return _decimals_from_step(info.get("stepSize", 6))

    def min_notional(self, symbol: str) -> float:
        info = self.markets.get(symbol) or {}
        try:
            return float(info.get("minNotional") or 1.0)
        except (TypeError, ValueError):
            return 1.0

    def min_qty(self, symbol: str) -> float:
        info = self.markets.get(symbol) or {}
        try:
            return float(info.get("minQty") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def round_quantity(self, symbol: str, quantity: float) -> float:
        p = self.quantity_precision(symbol)
        return float(f"{quantity:.{p}f}")

    def place_order(self, symbol, side, quantity, price=None):
        qty = self.round_quantity(symbol, quantity)
        info = self.markets.get(symbol)
        if info and not info.get("isMarketTypeEnable", True):
            return {"ok": False, "error": f"{symbol} does not allow MARKET orders"}
        if qty <= self.min_qty(symbol):
            return {"ok": False, "error": f"quantity {qty} <= minQty"}
        if price is None:
            try:
                px = self.client.last_price(symbol)
            except WallexClientError as e:
                return {"ok": False, "error": f"no price: {e}"}
        else:
            px = float(price)
        if px <= 0:
            return {"ok": False, "error": f"invalid price {px}"}
        if px * qty < self.min_notional(symbol):
            return {"ok": False, "error": f"notional {px * qty:.4f} < minNotional "
                                          f"{self.min_notional(symbol)}"}
        if self.dry_run:
            # Paper fill uses the caller-supplied bar price (signal/fill parity
            # with shadow replay); only fall back to the live tick if absent.
            fill_px = px
            self.paper.apply_fill(symbol, side, qty, fill_px)
            return {"ok": True, "dry_run": True, "symbol": symbol, "side": side,
                    "quantity": qty, "price": fill_px,
                    "portfolio_usdt_left": self.paper.usdt}
        result = self.client.place_order(symbol, side, qty,
                                         price=price if price else None)
        return {"ok": True, "dry_run": False, "symbol": symbol, "side": side,
                "quantity": qty, **result}

    def cancel_order(self, order_id):
        if self.dry_run:
            return {"ok": True, "dry_run": True, "canceled": order_id}
        return self.client.cancel_order(order_id)

    def get_balance(self) -> dict:
        if self.dry_run:
            return {"USDT": self.paper.usdt, **self.paper.holdings}
        return self.client.get_balances()

    def free_usdt(self) -> float:
        return float(self.get_balance().get("USDT", 0.0))

    def to_json(self) -> str:
        return json.dumps({"dry_run": self.dry_run,
                           "paper": {"usdt": self.paper.usdt,
                                     "holdings": self.paper.holdings,
                                     "realized_pnl": self.paper.realized_pnl}})
