"""Select the best portable phase-1 strategy for live/shadow deployment.

    python scripts/select_strategy.py --direction long
    python scripts/select_strategy.py --direction short
    python scripts/select_strategy.py --direction both

Scans results/{direction}/strategy_*.json (the NSGA-II outputs, both the small
summary files and the large files with trade lists), deduplicates identical
rule-sets, rejects strategies that use features live cannot compute, and ranks
the rest by a risk-adjusted out-of-sample score:

    score = (test_pnl / max(test_dd, 0.5)) * sqrt(test_pf) * size_factor

with hard gates on train/test profitability, trade count, PF and drawdown.
Writes results/fits/{direction}/ruleset.json (deploy format) plus a
strategy_selection.json audit trail.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from src.features import registry
from src.live.evaluator import computable_feature_names


def _canonical_rules(rules):
    return tuple(sorted(
        (tuple(sorted(r["conditions"])), float(r["tp"]), float(r["sl"]),
         float(r["capital_pct"])) for r in rules))


def _iter_candidates(direction):
    seen = set()
    for path in sorted(glob.glob(f"results/{direction}/strategy_*.json")):
        if os.path.getsize(path) > 50_000_000:
            continue
        try:
            with open(path, "r", encoding="utf8") as f:
                d = json.load(f)
        except Exception:
            continue
        strategy = d.get("strategy") or {}
        rules = strategy.get("rules_set") or d.get("rules_set") or []
        if not rules or "test" not in d or "train" not in d:
            continue
        rules = [{**r,
                  "conditions": list(r.get("conditions") or []),
                  "tp": float(r.get("tp", 1.0)),
                  "sl": float(r.get("sl", 1.0)),
                  "capital_pct": float(r.get("capital_pct", 5.0))}
                 for r in rules]
        key = _canonical_rules(rules)
        if key in seen:
            continue
        seen.add(key)
        yield path, d, rules


def _missing_features(rules, computable):
    feats = set()
    for r in rules:
        for c in r["conditions"]:
            feats.add(c.split("]")[0].strip("[]"))
    return sorted(feats - computable)


def _score(train, test):
    trades = max(int(test.get("trades", 0)), 1)
    size = min(1.0, trades / 300.0) ** 0.15
    return ((float(test.get("pnl", 0)) / max(float(test.get("dd", 0)), 0.5))
            * (max(float(test.get("pf", 0)), 0.0) ** 0.5) * size)


def select(direction, min_trades, min_pf, max_dd):
    cfg = load_config(direction=direction)
    registry.register_horizon_features(cfg.horizons)
    computable = computable_feature_names()
    rows = []
    for path, d, rules in _iter_candidates(direction):
        tr, te = d["train"], d["test"]
        missing = _missing_features(rules, computable)
        reasons = []
        if missing:
            reasons.append(f"features not computable live: {missing}")
        if float(tr.get("pnl", 0)) <= 0:
            reasons.append("train pnl <= 0")
        if float(te.get("pnl", 0)) <= 0:
            reasons.append("test pnl <= 0")
        if int(te.get("trades", 0)) < min_trades:
            reasons.append(f"test trades < {min_trades}")
        if float(te.get("pf", 0)) < min_pf:
            reasons.append(f"test pf < {min_pf}")
        if float(te.get("dd", 999)) > max_dd:
            reasons.append(f"test dd > {max_dd}")
        row = {"path": path, "rules": rules, "train": tr, "test": te,
               "missing_features": missing, "reasons": reasons,
               "score": _score(tr, te) if not reasons else None}
        rows.append(row)
    rows.sort(key=lambda r: -(r["score"] or -1e9))
    if not rows or rows[0]["score"] is None:
        raise SystemExit(f"No portable {direction} strategy passed the gates:\n" +
                         "\n".join(f"{os.path.basename(r['path'])}: {r['reasons']}"
                                   for r in rows[:10]))
    best = rows[0]
    out_dir = f"results/fits/{direction}"
    os.makedirs(out_dir, exist_ok=True)
    ruleset = {
        "direction": direction,
        "source": os.path.basename(best["path"]),
        "selection_score": round(best["score"], 4),
        "train": best["train"], "test": best["test"],
        "rules": [
            {"rule_id": i,
             "name": f"r{i}_{'+'.join(c.split(']')[0].strip('[]') for c in r['conditions'][:2])}",
             "conditions": r["conditions"],
             "tp": r["tp"], "sl": r["sl"],
             "capital_pct": r["capital_pct"]}
            for i, r in enumerate(best["rules"], start=1)],
    }
    with open(os.path.join(out_dir, "ruleset.json"), "w", encoding="utf8") as f:
        json.dump(ruleset, f, indent=1, default=str)
    audit = {
        "direction": direction,
        "selected": os.path.basename(best["path"]),
        "score": round(best["score"], 4),
        "gates": {"min_trades": min_trades, "min_pf": min_pf, "max_dd": max_dd},
        "ranking": [
            {"path": os.path.basename(r["path"]),
             "score": round(r["score"], 4) if r["score"] is not None else None,
             "test_pnl": r["test"].get("pnl"), "test_pf": r["test"].get("pf"),
             "test_wr": r["test"].get("wr"), "test_dd": r["test"].get("dd"),
             "test_trades": r["test"].get("trades"),
             "reasons": r["reasons"]}
            for r in rows[:15]],
    }
    with open(os.path.join(out_dir, "strategy_selection.json"), "w",
              encoding="utf8") as f:
        json.dump(audit, f, indent=1, default=str)
    print(f"✅ {direction}: selected {os.path.basename(best['path'])} "
          f"score={best['score']:.3f} test_pnl={best['test'].get('pnl'):.2f} "
          f"pf={best['test'].get('pf'):.2f} wr={best['test'].get('wr'):.1f} "
          f"dd={best['test'].get('dd'):.1f} trades={best['test'].get('trades')}")
    return ruleset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="both",
                    choices=["long", "short", "both"])
    ap.add_argument("--min-trades", type=int, default=150)
    ap.add_argument("--min-pf", type=float, default=1.10)
    ap.add_argument("--max-dd", type=float, default=12.0)
    args = ap.parse_args()
    for d in (["long", "short"] if args.direction == "both"
              else [args.direction]):
        select(d, args.min_trades, args.min_pf, args.max_dd)


if __name__ == "__main__":
    main()
