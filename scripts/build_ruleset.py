"""Build the deployable ruleset.json from phase-1 strategy fits.

Scans results/fits/{direction}/strategy_*.json and picks the best strategies
requiring BOTH train-positive and test-positive performance (no test-only
selection), ranked by min(train_pnl, test_pnl). Writes ruleset.json:

    python scripts/build_ruleset.py --direction long
Options: --top N (keep N best rules), --min-trades, --max-dd.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.fit_package import load_ruleset

MIN_TRADES = 50
MAX_DD = 10.0


def pick_strategy(fits_dir: str, min_trades: int, max_dd: float):
    cands = []
    for path in sorted(glob.glob(os.path.join(fits_dir, "strategy_*.json"))):
        try:
            blob = json.load(open(path))
        except Exception:
            continue
        tgt = blob.get("strategy") or blob
        test = blob.get("test") or {}
        train = blob.get("train") or {}
        rs = tgt.get("rules_set") or tgt.get("rules")
        if not rs:
            continue
        test_trades = test.get("trades", 0)
        test_dd = test.get("dd", 0.0)
        test_pnl = test.get("pnl", -999)
        train_trades = train.get("trades", 0)
        train_dd = train.get("dd", 0.0)
        train_pnl = train.get("pnl", -999)
        # both splits must be profitable with enough evidence on each side
        if test_trades >= min_trades and train_trades >= min_trades \
                and test_dd <= max_dd and train_dd <= max_dd \
                and test_pnl > 0 and train_pnl > 0:
            score = min(train_pnl, test_pnl)
            cands.append((score, train_pnl, test_pnl, train_trades,
                          test_trades, train_dd, test_dd, path, rs))
    return sorted(cands, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="long", choices=["long", "short"])
    ap.add_argument("--top", type=int, default=3, help="keep N best rules")
    ap.add_argument("--min-trades", type=int, default=MIN_TRADES)
    ap.add_argument("--max-dd", type=float, default=MAX_DD)
    ap.add_argument("--force", action="store_true",
                    help="write ruleset even when no strategy passes the filter")
    args = ap.parse_args()
    fits = os.path.join("results", "fits", args.direction)
    cands = pick_strategy(fits, args.min_trades, args.max_dd)
    if not cands:
        print("No strategy passed the filter (train & test both positive, "
              f"trades>={args.min_trades}, dd<={args.max_dd}).")
        if not args.force:
            print("Refusing to overwrite ruleset.json (use --force to override).")
            return 1
    rules = []
    for score, train_pnl, test_pnl, train_trades, test_trades, train_dd, test_dd, path, rules_set in cands[:args.top]:
        blob = json.load(open(path))
        for r in rules_set:
            rules.append({
                "rule_id": len(rules),
                "name": os.path.basename(path).replace(".json", ""),
                "conditions": r["conditions"],
                "tp": r.get("tp", 1.0), "sl": r.get("sl", 1.0),
                "capital_pct": r.get("capital_pct", 5.0),
                "_train": {"pnl": train_pnl, "trades": train_trades, "dd": train_dd},
                "_test": {"pnl": test_pnl, "trades": test_trades, "dd": test_dd},
                "_rank_score_min_split": score,
            })
    out = os.path.join(fits, "ruleset.json")
    payload = {"direction": args.direction, "rules": rules}
    json.dump(payload, open(out, "w"), indent=1)
    print(f"Wrote {out}: {len(rules)} rules")
    rs = load_ruleset(out)
    print(f"validated: {len(rs['rules'])} rules, direction={rs['direction']}")
    for c in cands[:args.top]:
        print(f"  {os.path.basename(c[7])}: train_pnl={c[1]:.2f}% "
              f"test_pnl={c[2]:.2f}% trades(train/test)={c[3]}/{c[4]} "
              f"dd(train/test)={c[5]:.2f}/{c[6]:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
