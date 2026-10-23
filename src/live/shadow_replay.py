"""Offline shadow replay: push the last N days of cached bars through the
fitted ruleset and simulate trades with entry at the NEXT open.

    python main_live.py --direction long --shadow-replay 14 \
        --symbols BTCUSDT ETHUSDT

Parity choices:
  - entry at the NEXT bar open after a signal (what a real post-close order gets)
  - at most ONE signal per bar, first matching rule wins (phase-1 semantics)
  - TP/SL intrabar with conservative SL-first tie-break and gap fills at open
  - 288-bar time exit; direction-aware 0.2% fee + 0.05% slippage per side
  - overlapping positions are allowed up to the exposure/capital cap, like the
    phase-1 portfolio simulator (set max_positions_per_symbol to restrict)
  - the feature frame includes the full 60-day warm-up BEFORE the replay window
"""
import logging
import time
from typing import Dict, Optional

import pandas as pd

from src.live.shadow import ShadowBroker

logger = logging.getLogger(__name__)

BARS_PER_DAY = 288
EXIT_COLS = ["datetime", "open", "high", "low", "close", "volume"]


class ShadowReplay:

    def __init__(self, trader, days: float, initial_usdt: float = 1000.0,
                 log_dir=None, max_positions_per_symbol: int = 1000):
        self.trader = trader
        self.days = days
        self.broker = ShadowBroker(
            initial_usdt=initial_usdt,
            direction=trader.direction,
            max_positions_per_symbol=max_positions_per_symbol,
            log_dir=log_dir or f"results/logs/shadow_{trader.direction}")

    @staticmethod
    def _mark(pos, bar):
        pos["bars_held"] = int(pos.get("bars_held", 0)) + 1
        pos["last_checked_bar"] = str(bar["datetime"])

    def run(self, symbols, days: Optional[float] = None) -> Dict:
        broker = self.broker
        days = days or self.days
        t0 = time.time()
        total_rows, total_signals = 0, 0
        warmup = self.trader.evaluator.require_bars
        replay_bars = int(days * BARS_PER_DAY)

        for sym in symbols:
            df = self.trader.evaluator.features_df(
                sym, tail_bars=warmup + replay_bars + 2)
            df = df.tail(replay_bars).reset_index(drop=True)
            n = len(df)
            total_rows += n
            broker.positions = []          # fresh book per symbol
            pending: Optional[Dict] = None
            pending_bar: Optional[str] = None

            for i in range(n):
                row = df.iloc[i]
                bar = row[EXIT_COLS].to_frame().T

                # 1) intrabar TP/SL for every open position
                for pos in list(broker.positions):
                    rec = broker._check_exit(pos, bar)
                    if rec:
                        broker.positions.remove(pos)
                    else:
                        self._mark(pos, row)

                # 2) execute the pending entry at THIS bar's open
                if pending is not None:
                    pos = broker.submit(pending, float(row["open"]), pending_bar)
                    pending, pending_bar = None, None
                    if pos is not None:
                        rec = broker._check_exit(pos, bar)
                        if rec:
                            broker.positions.remove(pos)
                        else:
                            self._mark(pos, row)

                # 3) one signal per closed bar -> pending entry next bar
                codes = self.trader.evaluator.row_codes_row(df, i)
                if not codes or pending is not None:
                    continue
                trig = self.trader.evaluator.rule_engine.evaluate_row(codes, sym)
                if trig:
                    pending = trig[0]
                    pending["symbol"] = sym
                    pending["direction"] = self.trader.direction
                    pending_bar = str(row["datetime"])
                    total_signals += 1
                    broker.log_signals([pending], pending_bar, "replay_entry")

            # Force-close at end of data, like the phase-1 engine.
            if broker.positions:
                broker.close_open({sym: float(df.iloc[-1]["close"])},
                                  reason="end_of_data")

        stats = broker.snapshot()
        stats["replay"] = {"days": days, "symbols": symbols,
                           "bars": total_rows, "signals": total_signals,
                           "elapsed_s": round(time.time() - t0, 1),
                           "warmup_bars": warmup,
                           "max_positions_per_symbol":
                               broker.max_positions_per_symbol}
        logger.info("Shadow replay %s: %d bars | %d signals | %d closed trades "
                    "| equity %.2f (%.2f%%) in %.1fs",
                    self.trader.direction, total_rows, total_signals,
                    stats["closed_trades"], stats["equity"],
                    stats["total_return_pct"], stats["replay"]["elapsed_s"])
        return stats

    # ---------------- reporting ----------------

    @staticmethod
    def report(stats: Dict) -> str:
        lines = ["=" * 78,
                 f"SHADOW STATS [{stats.get('direction','?')}]  "
                 f"({stats['closed_trades']} closed trades, "
                 f"{stats['total_return_pct']:+.2f}% equity, "
                 f"PF={stats.get('profit_factor')}, "
                 f"WR={stats.get('win_rate')}, "
                 f"maxDD={stats.get('max_drawdown_pct')}%, "
                 f"Sharpe/trade={stats.get('sharpe_trade')})",
                 "-" * 78]
        for k, a in sorted(stats["rules"].items(),
                           key=lambda kv: -kv[1]["trades"]):
            lines.append(
                f"{k:14s} {str(a['rule_name']):14s} trades={a['trades']:4d} "
                f"win%={a['win_rate']*100:5.1f} PF={str(a['profit_factor']):>5s} "
                f"avgPnL={a['avg_pnl']:+6.2f} hold={a['avg_hold_bars']} bars "
                f"ret={a['total_return_pct']:+6.2f}%")
        lines.append("=" * 78)
        return "\n".join(lines)
