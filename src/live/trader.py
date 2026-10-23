"""Phase 2 live trader: bar-aligned loop that turns fitted rules into Wallex orders.

Per 5m cycle (aligned to bar close + small fetch buffer):

    1. refresh candle caches for all symbols (threaded REST, latency tracked)
    2. evaluate rules on each symbol's last CLOSED bar
    3. manage open positions: every missed bar is replayed intrabar (TP/SL +
       288-bar time exit), so a connection outage cannot hide a stop
    4. size + place MARKET entries per rule capital_pct of free USDT
    5. log everything (signals.jsonl / trades.jsonl / cycles.jsonl)

Safety / outage policy:
    - real orders require dry_run=False AND a WALLEX_API_KEY
    - symbols whose feed is more than 3 bars stale are never traded; entries
      are skipped and the condition is logged
    - market metadata is disk-cached and the loop enters a degraded mode when
      Wallex is unreachable, retrying with backoff and recovering in place
    - exit orders that fail keep the position open and are retried (never
      mark an exchange-held position closed)
"""
import logging
import os
import signal
import time
from typing import Dict, List, Optional

import pandas as pd

from src.execution.wallex import WallexClient, WallexExecutor
from src.core.discretizer import normalize_direction
from src.live.evaluator import (_last_closed_candle_time, default_warmup_bars)
from src.live.fit_package import load_discretizer, load_ruleset
from src.live.logger import log_cycle, log_signal, log_trade, set_log_dir
from src.live.state import LiveState

logger = logging.getLogger(__name__)

BAR_SECONDS = 300
FETCH_BUFFER = 8          # seconds after bar close before evaluating
MAX_EXPOSURE = 0.9        # max fraction of equity allocated simultaneously
MAX_HOLD_BARS = 288       # time exit parity with the backtester (24h on 5m)
STALE_BARS = 3            # do not trade a symbol whose feed is older than this
MAX_BACKOFF_S = 60


def _entry_side(direction: str) -> str:
    return "SELL" if direction == "short" else "BUY"


def _closed_bars(df: pd.DataFrame, cutoff: pd.Timestamp,
                 tail: int = MAX_HOLD_BARS + 2) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df[df["datetime"] <= cutoff]
    return out.tail(tail).reset_index(drop=True)


class LiveTrader:

    def __init__(self, ruleset_path: str, package_path: str,
                 api_key: Optional[str] = None, dry_run: bool = True,
                 direction: Optional[str] = None, symbols: List[str] = None,
                 max_symbols: int = 12, paper_usdt: float = 1000.0,
                 store: str = "auto", shadow: bool = False,
                 state_path: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 warmup_bars: Optional[int] = None,
                 history_bars: Optional[int] = None,
                 client: Optional[WallexClient] = None,
                 source=None,
                 symbols_resolved: Optional[List[str]] = None):
        self.ruleset = load_ruleset(ruleset_path)
        resolved_direction = normalize_direction(
            direction or self.ruleset.get("direction", "long"))
        self.ruleset["direction"] = resolved_direction
        self.direction = resolved_direction
        if self.direction == "short" and not dry_run:
            logger.warning(
                "direction=short with --live: ensure the account/venue actually "
                "supports short selling — entries are SELL / exits are BUY.")
        self.disc = load_discretizer(package_path)
        self.dry_run = dry_run
        self.shadow_mode = shadow
        # warmup_bars = feature warm-up only (decision bar needs 2*30d bars).
        # history_bars = warm-up + replay window; the store must cache this much.
        self.warmup_bars = int(warmup_bars or default_warmup_bars())
        self.history_bars = int(history_bars or (self.warmup_bars + 288))

        if log_dir:
            set_log_dir(log_dir)

        self.client = client or WallexClient(api_key=api_key)
        self.degraded = False
        try:
            markets = self.client.get_markets()
        except Exception as e:  # no cache + no network
            markets = {}
            self.degraded = True
            logger.error("Market metadata unavailable (%s) — starting in "
                         "degraded mode; no entries until connectivity returns", e)
        self.executor = WallexExecutor(self.client, dry_run=dry_run,
                                       paper_usdt=paper_usdt, markets=markets)

        from src.data.live_store import LiveCandleStore
        self.source = source or LiveCandleStore(
            self.client, force_csv=(store == "csv"),
            warmup_bars=self.history_bars)

        if shadow:
            from src.live.shadow import ShadowBroker
            self.broker = ShadowBroker(initial_usdt=paper_usdt,
                                       direction=self.direction,
                                       log_dir=log_dir)
        else:
            self.broker = None

        default_state = (f"results/state/shadow_{self.direction}.json"
                         if shadow else
                         f"results/live_state_{self.direction}.json")
        self.state = LiveState(state_path or default_state)

        from src.live.evaluator import LiveEvaluator
        self.evaluator = LiveEvaluator(self.disc, self.source,
                                       require_bars=self.warmup_bars)
        self.evaluator.set_rules(self.ruleset)

        self.symbols = (symbols_resolved or
                        self._resolve_symbols(symbols, max_symbols))
        self.triggered_bars: Dict[str, str] = {}
        self._stale_logged: Dict[str, float] = {}
        self._stop = False
        self.health = {"connected": not self.degraded,
                       "last_cycle_ts": None,
                       "consecutive_failures": 0}
        log_trade({"event": "startup", "dry_run": dry_run,
                   "mode": "shadow" if shadow else "live",
                   "direction": self.direction,
                   "warmup_bars": self.warmup_bars,
                   "history_bars": self.history_bars,
                   "rules": len(self.ruleset["rules"]),
                   "skipped_rules": len(self.evaluator.rule_engine.skipped_rules),
                   "degraded": self.degraded,
                   "state_path": str(self.state.path),
                   "symbols": self.symbols})

    # ---------------- symbol selection ----------------

    def _fallback_symbols(self, max_symbols: int) -> List[str]:
        try:
            from config import load_config
            configured = [s for s in load_config().symbols if s.endswith("USDT")]
        except Exception:
            configured = []
        configured = configured or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
        logger.warning("Using configured symbol fallback: %s", configured)
        return configured[:max_symbols]

    def _resolve_symbols(self, symbols: Optional[List[str]],
                         max_symbols: int) -> List[str]:
        if symbols and symbols != ["auto"]:
            return [s.upper() for s in symbols]
        try:
            ranked = self.client.usdt_markets_by_volume()
        except Exception as e:
            logger.error("Could not rank markets by volume (%s)", e)
            ranked = []
        if not ranked:
            return self._fallback_symbols(max_symbols)
        chosen = ranked[:max_symbols]
        if "BTCUSDT" not in chosen and "BTCUSDT" in ranked:
            chosen = ["BTCUSDT"] + chosen[:-1]
        logger.info("Symbols (auto, by 24h USDT volume): %s", chosen)
        return chosen

    def _refresh_symbols_if_degraded(self):
        """After connectivity returns, refresh metadata and resolve symbols."""
        if not self.degraded:
            return
        try:
            markets = self.client.get_markets(force=True)
            if not markets:
                return
            self.executor.markets = markets
            self.degraded = False
            if self.symbols[:1] == ["auto"] or not self.symbols:
                self.symbols = self._resolve_symbols(["auto"], len(self.symbols) or 12)
            logger.info("Connectivity restored — market metadata refreshed, "
                        "degraded mode OFF")
        except Exception as e:
            logger.warning("Still disconnected: %s", e)

    # ---------------- positions ----------------

    def _check_exits(self, bars: Dict[str, pd.DataFrame]):
        """Replay every new closed bar since the last check, per position.

        Exits are intrabar TP/SL with conservative stop-first tie-break and gap
        fills at the open (same rules as ShadowBroker). Replaying missed bars
        means a 1h outage cannot skip a stop that was hit in between.
        Positions older than MAX_HOLD_BARS are closed at the bar close.
        """
        now = time.time()
        for pos in list(self.state.data["positions"]):
            sym = pos["symbol"]
            df = bars.get(sym)
            if df is None or df.empty:
                # No bars at all for a long time: wall-clock safety exit.
                if now - float(pos.get("opened_at", now)) >= MAX_HOLD_BARS * BAR_SECONDS:
                    px = self._current_price(sym) or pos["entry_price"]
                    self._exit(pos, px, "time_exit")
                continue

            last = pos.get("last_checked_bar")
            if last is None:
                sub = df.tail(1)
            else:
                sub = df[df["datetime"] > pd.Timestamp(last)]
            if sub.empty:
                continue

            attempted = False
            for _, row in sub.iterrows():
                pos["bars_held"] = int(pos.get("bars_held", 0)) + 1
                if pos["bars_held"] >= MAX_HOLD_BARS:
                    ok = self._exit(pos, float(row["close"]), "time_exit")
                    attempted = True
                    if not ok:
                        break
                    break
                rec, attempted = self._intrabar_exit(pos, row)
                if attempted:
                    break
                pos["last_checked_bar"] = str(row["datetime"])
            if not attempted and pos in self.state.data["positions"]:
                pos["last_checked_bar"] = str(sub.iloc[-1]["datetime"])
                self.state.update_position(pos)
            elif not attempted:
                continue
            # On success _exit already persisted/removed the position. On a
            # failed exit we intentionally do NOT advance last_checked_bar so
            # the same bar is retried next cycle.

    def _intrabar_exit(self, pos: Dict, row) -> tuple:
        """Returns (record|None, attempted). attempted=True when an exit level
        was touched; the caller stops advancing bars either way."""
        direction = pos["direction"]
        entry, tp, sl = float(pos["entry_price"]), float(pos["tp"]), float(pos["sl"])
        o, h, lo = float(row["open"]), float(row["high"]), float(row["low"])
        if direction == "long":
            tp_px, sl_px = entry * (1 + tp / 100), entry * (1 - sl / 100)
            if lo <= sl_px:
                px, reason = (min(o, sl_px) if o < sl_px else sl_px), "stop_loss"
            elif h >= tp_px:
                px, reason = (max(o, tp_px) if o > tp_px else tp_px), "take_profit"
            else:
                return None, False
        else:
            tp_px, sl_px = entry * (1 - tp / 100), entry * (1 + sl / 100)
            if h >= sl_px:
                px, reason = (max(o, sl_px) if o > sl_px else sl_px), "stop_loss"
            elif lo <= tp_px:
                px, reason = (min(o, tp_px) if o < tp_px else tp_px), "take_profit"
            else:
                return None, False
        ok = self._exit(pos, px, reason)
        return ({"reason": reason, "exit_price": px} if ok else None), True

    def _exit(self, pos: Dict, px: float, reason: str) -> bool:
        qty = pos["quantity"]
        fill = self.executor.place_order(
            pos["symbol"], "SELL" if pos["direction"] == "long" else "BUY",
            qty, price=(px if self.dry_run else None))
        if not fill.get("ok"):
            # Keep the position open and retry — the exchange may still hold it.
            log_signal({**self._signal_record(
                {"symbol": pos["symbol"], "bar_datetime": None,
                 "rule_id": pos.get("rule_id"), "rule_name": pos.get("rule"),
                 "conditions": pos.get("conditions", []),
                 "condition_detail": [], "tp": pos.get("tp"),
                 "sl": pos.get("sl"), "capital_pct": pos.get("capital_pct")},
                px),
                "decision": "exit_order_failed", "reason": reason,
                "error": fill.get("error"), "position_id": pos["position_id"]})
            logger.error("Exit order failed for %s (%s): %s — will retry",
                         pos["symbol"], reason, fill.get("error"))
            return False
        actual_px = fill.get("price") or px
        rec = {"event": "position_exit", "reason": reason, "fill": fill,
               "exit_price": px,
               "filled_price": actual_px,
               "pnl_pct": ((actual_px - pos["entry_price"]) / pos["entry_price"] * 100
                           if pos["direction"] == "long" else
                           (pos["entry_price"] - actual_px) / pos["entry_price"] * 100)}
        self.state.close_position(pos["position_id"], rec)
        log_trade({"position_id": pos["position_id"], "symbol": pos["symbol"],
                   **rec, "dry_run": self.dry_run})
        return True

    # ---------------- entries & sizing ----------------

    def _open_notional(self) -> float:
        total = 0.0
        for pos in self.state.data["positions"]:
            px = self._current_price(pos["symbol"])
            if px:
                total += abs(pos["quantity"]) * px
        return total

    def _try_entry(self, trigger: Dict, price: float) -> Optional[Dict]:
        sym = trigger["symbol"]
        rule_name = trigger.get("name") or f"rule_{trigger.get('rule_id')}"
        if self.degraded:
            log_signal({**self._signal_record(trigger, price),
                        "decision": "blocked", "reason": "disconnected"})
            return None
        if self.state.open_positions_for(sym):
            return None
        open_notional = self._open_notional()
        free = self.executor.free_usdt()
        alloc = free * (trigger["capital_pct"] / 100.0)
        equity = max(free + open_notional, 1e-9)
        min_notional = self.executor.min_notional(sym)
        if alloc <= 0:
            reason = "no free USDT"
        elif free < min_notional:
            reason = "free USDT below minNotional"
        elif (open_notional + alloc) / equity > MAX_EXPOSURE:
            reason = "allocation exceeds max exposure cap"
        elif alloc < min_notional:
            reason = f"allocation {alloc:.4f} below minNotional {min_notional}"
        else:
            reason = None
        if reason:
            log_signal({**self._signal_record(trigger, price),
                        "decision": "blocked", "reason": reason})
            return None
        qty = self.executor.round_quantity(sym, alloc / price)
        if qty <= 0:
            log_signal({**self._signal_record(trigger, price),
                        "decision": "blocked", "reason": "quantity rounds to zero"})
            return None
        side = _entry_side(self.direction)
        fill = self.executor.place_order(
            sym, side, qty, price=(price if self.dry_run else None))
        rec = self._signal_record(trigger, price)
        if not fill.get("ok"):
            log_signal({**rec, "decision": "order_rejected",
                        "error": fill.get("error")})
            return None
        fill_price = fill.get("price") or price
        pos = self.state.open_position(
            symbol=sym, direction=self.direction,
            rule=rule_name, rule_id=trigger["rule_id"], tp=trigger["tp"],
            sl=trigger["sl"], capital_pct=trigger["capital_pct"],
            entry_price=fill_price, quantity=fill["quantity"],
            bar_datetime=trigger["bar_datetime"], conditions=trigger["conditions"],
            last_checked_bar=trigger["bar_datetime"], bars_held=0)
        log_signal({**rec, "decision": "entered",
                    "quantity": fill["quantity"]})
        log_trade({"event": "position_open", "position_id": pos["position_id"],
                   "fill": fill, "dry_run": self.dry_run})
        return pos

    def _current_price(self, sym: str) -> Optional[float]:
        try:
            df = self.source.read_pair(sym, refresh=False)
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception:
            pass
        try:
            return float(self.client.last_price(sym))
        except Exception:
            return None

    def _signal_record(self, trigger, price) -> Dict:
        return {"event": "signal", "symbol": trigger["symbol"],
                "bar_datetime": trigger.get("bar_datetime"),
                "close": price, "rule_id": trigger.get("rule_id"),
                "rule_name": trigger.get("name"),
                "rule_text": " AND ".join(trigger["conditions"]),
                "conditions_detail": trigger["condition_detail"],
                "tp": trigger["tp"], "sl": trigger["sl"],
                "capital_pct": trigger["capital_pct"],
                "direction": self.direction,
                "dry_run": self.dry_run}

    # ---------------- cycle helpers ----------------

    def _read_bars(self, cutoff: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        out = {}
        for sym in self.symbols:
            try:
                df = self.source.read_pair(sym, refresh=False)
            except Exception as e:
                logger.warning("Read failed for %s: %s", sym, e)
                continue
            df = _closed_bars(df, cutoff)
            if df is not None and not df.empty:
                out[sym] = df
        return out

    def _is_stale(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> bool:
        last = pd.Timestamp(df["datetime"].iloc[-1])
        return (cutoff - last).total_seconds() > STALE_BARS * BAR_SECONDS

    def _note_stale(self, sym: str, df: pd.DataFrame, cutoff: pd.Timestamp):
        now = time.time()
        if now - self._stale_logged.get(sym, 0) > 600:
            self._stale_logged[sym] = now
            logger.warning("Stale feed %s: last closed bar %s, cutoff %s — "
                           "skipping entries (will recover automatically)",
                           sym, df["datetime"].iloc[-1], cutoff)

    def _prices(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        return {s: float(df["close"].iloc[-1]) for s, df in bars.items()
                if df is not None and not df.empty}

    # ---------------- shadow mode ----------------

    def run_shadow_cycle(self) -> dict:
        """Shadow mode: same refresh/eval, execution through ShadowBroker."""
        t0 = time.time()
        now = int(t0)
        refresh = self.source.refresh_symbols(self.symbols, max_workers=4)
        cutoff = _last_closed_candle_time()
        bars = self._read_bars(cutoff)
        self.broker.on_bars(bars)  # exits first (replays every missed bar)

        fired = skipped_stale = 0
        for sym in self.symbols:
            df = bars.get(sym)
            if df is None or df.empty:
                continue
            if self._is_stale(df, cutoff):
                self._note_stale(sym, df, cutoff)
                skipped_stale += 1
                continue
            try:
                triggers = self.evaluator.evaluate_symbol(sym)
            except Exception as e:
                logger.warning("Shadow eval failed for %s: %s", sym, e)
                continue
            for trig in triggers:
                pos = self.broker.submit(trig, float(trig["close"]),
                                         trig["bar_datetime"])
                self.broker.log_signals(
                    [trig], trig["bar_datetime"],
                    "shadow_entry" if pos else "shadow_skip_position_open")
                fired += bool(pos)
        stats = self.broker.snapshot()
        cycle = {"cycle_ts": now, "cycle_ms": round((time.time() - t0) * 1000),
                 "mode": "shadow", "direction": self.direction,
                 "symbols": len(self.symbols),
                 "refresh_errors": {s: r["error"] for s, r in refresh.items()
                                    if isinstance(r, dict) and r.get("error")},
                 "stale_symbols": skipped_stale,
                 "signals_fired": fired,
                 "shadow_equity": stats["equity"],
                 "closed_trades": stats["closed_trades"]}
        log_cycle(cycle)
        self.health.update({"last_cycle_ts": now,
                            "consecutive_failures": 0})
        return cycle

    # ---------------- paper / live mode ----------------

    def run_cycle(self) -> dict:
        t0 = time.time()
        now = int(t0)
        log_trade({"event": "cycle_start", "cycle_ts": now,
                   "direction": self.direction})
        refresh = self.source.refresh_symbols(self.symbols, max_workers=4)
        cutoff = _last_closed_candle_time()
        bars = self._read_bars(cutoff)
        self._check_exits(bars)

        fired = skipped_stale = 0
        for sym in self.symbols:
            df = bars.get(sym)
            if df is None or df.empty:
                continue
            if self._is_stale(df, cutoff):
                self._note_stale(sym, df, cutoff)
                skipped_stale += 1
                continue
            try:
                triggers = self.evaluator.evaluate_symbol(sym)
            except Exception as e:
                logger.warning("Eval failed for %s: %s", sym, e)
                continue
            for trig in triggers:
                bar_dt = pd.Timestamp(trig["bar_datetime"])
                bar_row = df[df["datetime"] == bar_dt]
                if bar_row.empty:
                    log_signal({**self._signal_record(trig, float(trig["close"])),
                                "decision": "stale_bar"})
                    continue
                price = float(bar_row["close"].iloc[-1])
                pos = self._try_entry(trig, price)
                fired += bool(pos)

        prices = self._prices(bars)
        cycle = {"cycle_ts": now,
                 "cycle_ms": round((time.time() - t0) * 1000),
                 "mode": "paper" if self.dry_run else "live",
                 "direction": self.direction,
                 "symbols": len(self.symbols),
                 "refresh": refresh,
                 "refresh_errors": {s: r["error"] for s, r in refresh.items()
                                    if isinstance(r, dict) and r.get("error")},
                 "stale_symbols": skipped_stale,
                 "signals_fired": fired,
                 "degraded": self.degraded,
                 "open_positions": len(self.state.data["positions"])}
        if self.dry_run:
            cycle["paper_equity"] = round(self.executor.paper.equity(prices), 4)
        log_cycle(cycle)
        log_trade({"event": "cycle_end", "cycle_ms": cycle["cycle_ms"],
                   "signals_fired": fired, "dry_run": self.dry_run,
                   "paper_equity": cycle.get("paper_equity"),
                   "refresh_errors": cycle["refresh_errors"]})
        self.health.update({"last_cycle_ts": now, "consecutive_failures": 0})
        return cycle

    # ---------------- loops ----------------

    def _install_signal_handlers(self):
        def _handler(signum, frame):
            logger.info("Signal %s received — finishing current cycle and "
                        "shutting down", signum)
            self._stop = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

    def run_forever(self, poll_interval_s: int = 15,
                    max_cycles: Optional[int] = None):
        logger.info("LiveTrader running: direction=%s symbols=%s rules=%d "
                    "dry_run=%s mode=%s", self.direction, self.symbols,
                    len(self.ruleset["rules"]), self.dry_run,
                    "shadow" if self.shadow_mode else "live")
        self._install_signal_handlers()
        last_bar = None
        cycles = 0
        backoff = 0
        while not self._stop:
            now = int(time.time())
            bar_now = now // BAR_SECONDS
            if bar_now != last_bar and (now % BAR_SECONDS) >= FETCH_BUFFER:
                try:
                    self._refresh_symbols_if_degraded()
                    if self.shadow_mode:
                        self.run_shadow_cycle()
                    else:
                        self.run_cycle()
                    last_bar = bar_now
                    cycles += 1
                    backoff = 0
                    self.health["consecutive_failures"] = 0
                    if max_cycles and cycles >= max_cycles:
                        logger.info("max_cycles=%d reached — exiting", max_cycles)
                        break
                except KeyboardInterrupt:
                    break
                except Exception:
                    self.health["consecutive_failures"] += 1
                    backoff = min(MAX_BACKOFF_S, max(5, backoff * 2 or 5))
                    logger.exception("Cycle crashed (failure #%d) — retrying in %ds",
                                     self.health["consecutive_failures"], backoff)
                    self._recover()
                    time.sleep(backoff)
                    continue
            time.sleep(poll_interval_s)
        logger.info("LiveTrader stopped after %d cycles", cycles)

    def _recover(self):
        """Best-effort recovery hooks after a failed cycle."""
        try:
            self.executor.refresh_markets()
        except Exception:
            pass
        try:
            self._refresh_symbols_if_degraded()
        except Exception:
            pass
