"""Shadow testing / paper trading (phase 2).

ShadowBroker simulates the FULL trade lifecycle without touching Wallex or the
paper portfolio, with deliberately conservative execution assumptions:

    entry  : next bar OPEN after the signal bar (offline replay) or the signal
             bar close in the live loop, plus direction-aware slippage + fee
    exits  : checked against every subsequent bar's high/low intrabar
             - LONG: stop hit first whenever both tp and sl lie inside one
               bar (conservative ordering; gap opens fill at the OPEN)
             - SHORT: mirror image (buy-stop above, buy-target below)
    costs  : 0.2% fee + 0.05% slippage per side, both included in the fills
    horizon: 288-bar time exit, matching the phase-1 backtester

Outage safety: `on_bars` replays EVERY bar since the position was last checked,
so a connection gap cannot skip a TP/SL that was touched while offline.

Outputs (per direction when configured via log_dir):
    shadow_signals.jsonl   every evaluated signal (explanation)
    shadow_trades.jsonl    every simulated fill (entry/exit)
    shadow_stats.json      rolling per-rule attribution + portfolio metrics
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.live.logger import LOG_DIR

logger = logging.getLogger(__name__)

FEE = 0.002               # Wallex flat 0.2% per side
SLIP_BPS = 5.0            # 0.05% per side
MAX_HOLD_BARS = 288       # time exit — parity with the backtester (24h on 5m)


class ShadowBroker:

    def __init__(self, initial_usdt: float = 1000.0, fee: float = FEE,
                 slip_bps: float = SLIP_BPS,
                 direction: str = "long",
                 max_positions_per_symbol: int = 1,
                 log_dir=None,
                 stats_path=None):
        self.initial_usdt = initial_usdt
        self.equity = initial_usdt
        self.peak_equity = initial_usdt
        self.fee = fee
        self.slip = slip_bps / 10_000.0
        self.direction = direction
        self.max_positions_per_symbol = int(max_positions_per_symbol)
        self.positions: List[Dict] = []
        self.closed: List[Dict] = []
        self.equity_curve: List[float] = [initial_usdt]
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stats_path = Path(stats_path) if stats_path else \
            self.log_dir / "shadow_stats.json"
        self.last_stats: Dict = {}

    # ---------------- costs / bookkeeping ----------------

    def _entry_fill(self, price: float, direction: str) -> float:
        """Effective entry fill: long buys pay up; short sells receive less."""
        if direction == "long":
            return price * (1 + self.slip) * (1 + self.fee)
        return price * (1 - self.slip) * (1 - self.fee)

    def _exit_fill(self, price: float, direction: str) -> float:
        """Effective exit fill: long sells receive less; short buys pay up."""
        if direction == "long":
            return price * (1 - self.slip) * (1 - self.fee)
        return price * (1 + self.slip) * (1 + self.fee)

    def _open_alloc(self) -> float:
        return sum(float(p.get("alloc", 0.0)) for p in self.positions)

    # ---------------- entries ----------------

    def submit(self, trigger: Dict, entry_price: float,
               bar_time: str) -> Optional[Dict]:
        """Open a shadow position if none open on the symbol and capital allows."""
        sym = trigger["symbol"]
        direction = trigger.get("direction", self.direction)
        open_sym = sum(1 for p in self.positions if p["symbol"] == sym)
        if open_sym >= self.max_positions_per_symbol:
            return None
        alloc = min(max(self.equity - self._open_alloc(), 0.0),
                    self.equity * (trigger.get("capital_pct", 5.0) / 100.0))
        if alloc <= 0:
            return None
        fill = self._entry_fill(float(entry_price), direction)
        qty = alloc / fill if fill > 0 else 0.0
        if qty <= 0:
            return None
        pos = {
            "shadow_id": f"{int(time.time() * 1e6)}-{len(self.closed)}-{sym}",
            "symbol": sym, "direction": direction,
            "rule_id": trigger.get("rule_id"),
            "rule_name": trigger.get("name", ""),
            "rule_text": " AND ".join(trigger.get("conditions", [])),
            "conditions_detail": trigger.get("condition_detail", []),
            "signal_bar": bar_time,
            "entry_bar": bar_time,
            "entry_price": float(entry_price),
            "entry_fill": float(fill),
            "entry_cost_price": float(fill),  # backward-compatible alias
            "quantity": qty, "alloc": alloc,
            "tp": float(trigger.get("tp", 1.0)),
            "sl": float(trigger.get("sl", 1.0)),
            "opened_cycle_ts": time.time(),
            "bars_held": 0,
            "last_checked_bar": str(bar_time),
        }
        self.positions.append(pos)
        self._log("shadow_trades.jsonl", {"event": "entry", **{
            k: pos[k] for k in ("shadow_id", "symbol", "direction", "rule_id",
                                "rule_name", "entry_price", "entry_fill",
                                "quantity", "tp", "sl", "signal_bar")}})
        return pos

    # ---------------- exits ----------------

    def on_bars(self, bars: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Replay every new closed bar for every open position.

        bars: {symbol: DataFrame} of recent closed bars (oldest -> newest).
        """
        done, still_open = [], []
        for pos in self.positions:
            df = bars.get(pos["symbol"])
            if df is None or df.empty:
                still_open.append(pos)
                continue
            last = pos.get("last_checked_bar")
            sub = df[df["datetime"] > pd.Timestamp(last)] if last else df.tail(1)
            if sub.empty:
                still_open.append(pos)
                continue
            rec, exit_attempted = None, False
            for _, row in sub.iterrows():
                pos["bars_held"] += 1
                rec = self._check_exit(pos, row.to_frame().T)
                if rec is not None:
                    exit_attempted = True
                    break
                pos["last_checked_bar"] = str(row["datetime"])
            if rec is None and not exit_attempted:
                still_open.append(pos)
            else:
                done.append(rec)
        self.positions = still_open
        return done

    def _check_exit(self, pos: Dict, df: pd.DataFrame) -> Optional[Dict]:
        """Intrabar TP/SL on one closed bar row; closes and records on trigger."""
        row = df.iloc[-1]
        direction = pos["direction"]
        entry = pos.get("entry_fill", pos.get("entry_cost_price"))
        o, h, lo = float(row["open"]), float(row["high"]), float(row["low"])
        c = float(row["close"])
        timed_out = pos.get("bars_held", 0) >= MAX_HOLD_BARS
        if direction == "long":
            tp_px = entry * (1 + pos["tp"] / 100)
            sl_px = entry * (1 - pos["sl"] / 100)
            if lo <= sl_px:
                px, reason = (min(o, sl_px) if o < sl_px else sl_px), "stop_loss"
            elif h >= tp_px:
                px, reason = (max(o, tp_px) if o > tp_px else tp_px), "take_profit"
            elif timed_out:
                px, reason = c, "time_exit"
            else:
                return None
            exit_px = self._exit_fill(px, "long")
            pnl = (exit_px - entry) * pos["quantity"]
            pnl_pct = (exit_px / entry - 1) * 100 if entry else 0.0
        else:
            tp_px = entry * (1 - pos["tp"] / 100)
            sl_px = entry * (1 + pos["sl"] / 100)
            if h >= sl_px:
                px, reason = (max(o, sl_px) if o > sl_px else sl_px), "stop_loss"
            elif lo <= tp_px:
                px, reason = (min(o, tp_px) if o < tp_px else tp_px), "take_profit"
            elif timed_out:
                px, reason = c, "time_exit"
            else:
                return None
            exit_px = self._exit_fill(px, "short")
            pnl = (entry - exit_px) * pos["quantity"]
            pnl_pct = (entry / exit_px - 1) * 100 if exit_px else 0.0

        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)
        rec = {**{k: pos[k] for k in ("shadow_id", "symbol", "direction",
                                      "rule_id", "rule_name", "rule_text",
                                      "entry_fill", "quantity", "bars_held")},
               "exit_bar": str(row["datetime"]), "exit_reason": reason,
               "exit_price": float(exit_px), "pnl": round(float(pnl), 4),
               "pnl_pct": round(float(pnl_pct), 4),
               "equity_after": round(self.equity, 4)}
        self.closed.append(rec)
        self._log("shadow_trades.jsonl", {"event": "exit", **rec})
        return rec

    def close_open(self, price_by_symbol: Dict[str, float],
                   reason: str = "end_of_data") -> List[Dict]:
        """Force-close remaining positions (end of replay / shutdown)."""
        done = []
        for pos in list(self.positions):
            px = price_by_symbol.get(pos["symbol"])
            if px is None:
                continue
            exit_px = self._exit_fill(float(px), pos["direction"])
            entry = pos.get("entry_fill", pos.get("entry_cost_price"))
            pnl = ((exit_px - entry) if pos["direction"] == "long"
                   else (entry - exit_px)) * pos["quantity"]
            self.equity += pnl
            rec = {**{k: pos[k] for k in ("shadow_id", "symbol", "direction",
                                          "rule_id", "rule_name", "rule_text",
                                          "entry_fill", "quantity", "bars_held")},
                   "exit_bar": pos.get("last_checked_bar"),
                   "exit_reason": reason, "exit_price": float(exit_px),
                   "pnl": round(float(pnl), 4),
                   "pnl_pct": round(float(pnl / max(pos["alloc"], 1e-9) * 100), 4),
                   "equity_after": round(self.equity, 4)}
            self.closed.append(rec)
            self.positions.remove(pos)
            done.append(rec)
            self.equity_curve.append(self.equity)
            self._log("shadow_trades.jsonl", {"event": "exit", **rec})
        return done

    def _log(self, name: str, record: Dict):
        path = self.log_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf8") as f:
            f.write(json.dumps({"ts": time.time(), **record}, default=str) + "\n")

    # ---------------- attribution ----------------

    def rule_stats(self) -> Dict[str, Dict]:
        agg: Dict[str, Dict] = {}
        for tr in self.closed:
            key = f"rule_{tr['rule_id']}"
            a = agg.setdefault(key, {"rule_name": tr["rule_name"],
                                     "rule_text": tr["rule_text"],
                                     "trades": 0, "wins": 0, "pnl_sum": 0.0,
                                     "pnl_wins": 0.0, "pnl_losses": 0.0,
                                     "bars": []})
            a["trades"] += 1
            a["wins"] += tr["pnl"] > 0
            a["pnl_sum"] += tr["pnl"]
            if tr["pnl"] > 0:
                a["pnl_wins"] += abs(tr["pnl"])
            else:
                a["pnl_losses"] += abs(tr["pnl"])
            a["bars"].append(tr["bars_held"])
        for a in agg.values():
            pf = (a["pnl_wins"] / a["pnl_losses"] if a["pnl_losses"] > 0
                  else (None if a["pnl_wins"] == 0 else 999.0))
            a.update({
                "win_rate": a["wins"] / a["trades"],
                "profit_factor": round(pf, 3) if pf is not None else None,
                "avg_pnl": round(a["pnl_sum"] / a["trades"], 4),
                "avg_hold_bars": round(float(np.mean(a["bars"])), 1)
                if a["bars"] else None,
                "total_return_pct":
                    round(a["pnl_sum"] / self.initial_usdt * 100, 3),
            })
            del a["bars"]
        return agg

    def _max_drawdown_pct(self) -> float:
        peak, dd = -float("inf"), 0.0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            if peak > 0:
                dd = max(dd, (peak - eq) / peak * 100)
        return round(dd, 3)

    def snapshot(self) -> Dict:
        pnls = [t["pnl"] for t in self.closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf = (sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0
              else (None if not wins else 999.0))
        rets = [t["pnl"] / max(t.get("alloc", self.initial_usdt), 1e-9)
                for t in self.closed]
        stats = {
            "ts": time.time(),
            "direction": self.direction,
            "initial_usdt": self.initial_usdt,
            "equity": round(self.equity, 4),
            "total_return_pct": round((self.equity / self.initial_usdt - 1) * 100, 3),
            "max_drawdown_pct": self._max_drawdown_pct(),
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed),
            "win_rate": round(len(wins) / len(pnls), 4) if pnls else None,
            "profit_factor": round(pf, 3) if pf is not None else None,
            "avg_trade_pct": round(float(np.mean(rets)) * 100, 4) if rets else None,
            "sharpe_trade": round(float(np.mean(rets) / np.std(rets)), 3)
            if len(rets) > 1 and np.std(rets) > 0 else None,
            "peak_equity": round(self.peak_equity, 4),
            "rules": self.rule_stats(),
        }
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.stats_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(stats, indent=1, default=str))
        tmp.replace(self.stats_path)
        self.last_stats = stats
        return stats

    def log_signals(self, triggers: List[Dict], bar_time: str, decision: str):
        for t in triggers:
            self._log("shadow_signals.jsonl", {
                "event": "signal", "decision": decision,
                "direction": t.get("direction", self.direction),
                "symbol": t["symbol"], "bar": bar_time,
                "rule_id": t.get("rule_id"), "rule_name": t.get("name"),
                "rule_text": " AND ".join(t["conditions"]),
                "conditions_detail": t.get("condition_detail"),
                "tp": t.get("tp"), "sl": t.get("sl"),
                "capital_pct": t.get("capital_pct"),
                "passed_filters": t.get("passed_filters", True),
            })
