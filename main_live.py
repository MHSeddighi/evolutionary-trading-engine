"""Phase 2 entrypoint: deploy the phase-1 strategy live on Wallex.

    python main_live.py --dry-run --once --direction long
    python main_live.py --shadow --once --direction both
    python main_live.py --direction long --shadow-replay 14
    python main_live.py --direction both --shadow-replay 30 --symbols BTCUSDT ETHUSDT
    WALLEX_API_KEY=... python main_live.py --direction long --live

Safety: paper/shadow are the default; --live requires WALLEX_API_KEY. The first
run backfills the feature warm-up (60 days for 30d features) unless --no-refresh
is passed with an already-warm cache.
"""
import argparse
import json
import logging
import os
import signal
import sys
import time

from src.live.trader import LiveTrader


def _paths(args, direction):
    ruleset = args.ruleset or f"results/fits/{direction}/ruleset.json"
    package = args.package or f"results/fits/{direction}/fit_package.pkl"
    if args.ruleset and args.direction == "both":
        raise SystemExit("--ruleset cannot be combined with --direction both")
    if args.package and args.direction == "both":
        raise SystemExit("--package cannot be combined with --direction both")
    return ruleset, package


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruleset", default=None,
                    help="default: results/fits/{direction}/ruleset.json")
    ap.add_argument("--package", default=None,
                    help="default: results/fits/{direction}/fit_package.pkl")
    ap.add_argument("--direction", default=None,
                    choices=["long", "short", "both"],
                    help="both = run long and short with isolated state/logs")
    ap.add_argument("--symbols", nargs="*", default=["auto"])
    ap.add_argument("--max-symbols", type=int, default=12)
    ap.add_argument("--paper-usdt", type=float, default=1000.0)
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="paper trading (default)")
    ap.add_argument("--once", action="store_true", help="single cycle and exit")
    ap.add_argument("--report", action="store_true",
                    help="print last signals with explanations and exit")
    ap.add_argument("--live", action="store_true",
                    help="REAL orders (--dry-run is the default)")
    ap.add_argument("--store", default="db", choices=["db", "csv"],
                    help="candle store: dockerized DB (default) or CSV cache")
    ap.add_argument("--shadow", action="store_true",
                    help="shadow mode: live-loop evaluation, trades simulated "
                         "(next-open entry, intrabar TP/SL, fee+slippage); "
                         "per-rule stats in results/logs/shadow_{direction}/")
    ap.add_argument("--shadow-replay", type=float, metavar="DAYS", default=None,
                    help="offline shadow test: replay the last DAYS of cached "
                         "bars and print per-rule attribution, then exit")
    ap.add_argument("--warmup-bars", type=int, default=None,
                    help="override cached-history requirement (default 2*max horizon+288)")
    ap.add_argument("--history-days", type=float, default=None,
                    help="replay window >= --shadow-replay; defaults to replay+1d")
    ap.add_argument("--no-refresh", action="store_true",
                    help="do not call Wallex (use the existing candle cache)")
    ap.add_argument("--state", default=None, help="override state JSON path")
    ap.add_argument("--log-dir", default=None, help="override log directory")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="stop after N bar cycles (shadow/paper soak tests)")
    ap.add_argument("--poll", type=int, default=15, help="loop poll seconds")
    args = ap.parse_args()

    dry_run = not args.live
    if not dry_run and not os.environ.get("WALLEX_API_KEY"):
        logging.error("--live requires WALLEX_API_KEY in the environment — refusing")
        sys.exit(2)

    directions = (["long", "short"] if args.direction == "both"
                  else [args.direction or "long"])

    if args.report:
        from src.live.logger import read_signals, set_log_dir
        if args.log_dir:
            set_log_dir(args.log_dir)
        elif len(directions) == 1:
            set_log_dir(f"results/logs/shadow_{directions[0]}"
                        if args.shadow else "results/logs")
        print(json.dumps(read_signals(50), indent=2, ensure_ascii=False))
        return

    from src.live.logger import setup_console
    setup_console(logging.INFO)

    from src.live.evaluator import default_warmup_bars
    from src.execution.wallex import WallexClient
    from src.data.live_store import LiveCandleStore

    # Two distinct requirements:
    #   feature_warmup = 2*max_horizon+288 (the decision bar needs 60 days)
    #   history_bars   = feature_warmup + replay window + margin
    feature_warmup = args.warmup_bars or default_warmup_bars()
    history_bars = feature_warmup + 288
    if args.shadow_replay:
        replay_days = args.shadow_replay
        history_days = args.history_days or (replay_days + 1)
        history_bars = max(history_bars,
                           int((default_warmup_bars() / 288 +
                                history_days + 1) * 288))

    client = WallexClient(api_key=os.environ.get("WALLEX_API_KEY"))
    shared_source = None
    if len(directions) > 1:
        shared_source = LiveCandleStore(client, force_csv=(args.store == "csv"),
                                        warmup_bars=history_bars)

    def make_trader(direction, symbols_resolved=None):
        ruleset, package = _paths(args, direction)
        log_dir = args.log_dir
        if log_dir is None and (args.shadow or args.shadow_replay or
                                len(directions) > 1):
            prefix = "shadow" if (args.shadow or args.shadow_replay) else "paper"
            log_dir = f"results/logs/{prefix}_{direction}"
        state = args.state
        if state is None and len(directions) > 1:
            prefix = "shadow" if (args.shadow or args.shadow_replay) else "paper"
            state = f"results/state/{prefix}_{direction}.json"
        return LiveTrader(
            ruleset_path=ruleset, package_path=package,
            api_key=os.environ.get("WALLEX_API_KEY"),
            dry_run=dry_run, direction=direction,
            symbols=args.symbols, max_symbols=args.max_symbols,
            paper_usdt=args.paper_usdt, store=args.store,
            shadow=(args.shadow or args.shadow_replay is not None),
            state_path=state, log_dir=log_dir, warmup_bars=feature_warmup,
            history_bars=history_bars,
            client=client, source=shared_source,
            symbols_resolved=symbols_resolved)

    traders = [make_trader(directions[0])]
    for d in directions[1:]:
        traders.append(make_trader(d, symbols_resolved=traders[0].symbols))

    symbols = traders[0].symbols
    if args.shadow_replay is not None:
        from src.live.shadow_replay import ShadowReplay
        if not args.no_refresh:
            logging.info("Ensuring %.0f days of candle history for %s "
                         "(feature warm-up %.0f days + replay) ...",
                         history_bars * 300 / 86400, symbols,
                         feature_warmup * 300 / 86400)
            traders[0].source.refresh_symbols(symbols, max_workers=4)
        combined = {"directions": {}}
        for trader in traders:
            reps = ShadowReplay(trader, args.shadow_replay,
                                initial_usdt=args.paper_usdt)
            stats = reps.run(symbols)
            print(ShadowReplay.report(stats))
            combined["directions"][trader.direction] = stats
        if len(traders) > 1:
            eq = sum(s["equity"] for s in combined["directions"].values())
            init = sum(s["initial_usdt"] for s in combined["directions"].values())
            combined.update({"combined_equity": round(eq, 2),
                             "combined_return_pct":
                                 round((eq / init - 1) * 100, 3)})
            path = "results/logs/shadow_both_stats.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf8") as f:
                json.dump(combined, f, indent=1, default=str)
            print(f"COMBINED: equity {eq:.2f} ({combined['combined_return_pct']:+.2f}%) "
                  f"-> {path}")
        return

    if args.once:
        for trader in traders:
            cycle = (trader.run_shadow_cycle() if trader.shadow_mode
                     else trader.run_cycle())
            print(json.dumps(cycle, indent=2, default=str))
        return

    # Continuous loop: either one trader, or both fired together each bar.
    if len(traders) == 1:
        traders[0].run_forever(poll_interval_s=args.poll,
                               max_cycles=args.max_cycles)
        return

    stop = {"flag": False}

    def _stop(_sig, _frame):
        stop["flag"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):
            pass
    logging.info("Both-direction loop: %s", [t.direction for t in traders])
    last_bar, cycles, backoff = None, 0, 0
    while not stop["flag"]:
        now = int(time.time())
        bar_now = now // 300
        if bar_now != last_bar and (now % 300) >= 8:
            try:
                for trader in traders:
                    if trader.shadow_mode:
                        trader.run_shadow_cycle()
                    else:
                        trader.run_cycle()
                last_bar = bar_now
                cycles += 1
                backoff = 0
                if args.max_cycles and cycles >= args.max_cycles:
                    break
            except Exception:
                logging.exception("Both-direction cycle failed — retrying")
                backoff = min(60, max(5, backoff * 2 or 5))
                time.sleep(backoff)
                continue
        time.sleep(args.poll)
    logging.info("Both-direction loop stopped after %d cycles", cycles)


if __name__ == "__main__":
    main()
