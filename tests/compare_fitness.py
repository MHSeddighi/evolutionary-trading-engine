"""Before/after fitness comparison on a representative stored strategy.

Old objectives (pre-refactor, reproduced here from the original formulas):
  REXP_old  = mean(trade_pnls) - 0.1*std(trade_pnls)                     [USD]
  RWP_old   = 0.5*mean(win_pnl) + 0.3*p10(windows) - 0.2*std(windows)    [USD]
              (30-trade windows, stride 1 = heavily overlapping)
  DIV_old   = unique_features + 0.5*|tps| + 0.5*|sls| - 2*repetition
New objectives: REXP, RWP, ROB as implemented in src/core/fitness.py.
Both are evaluated on the TRAIN split only — the test window is untouched.
"""
import json, time
import numpy as np

from src.features.builder import run
from src.data.loader import assemble
from src.core.backtest import CapitalManagedTradeSimulator
from src.core.fitness import BAD, _ulcer_pct
from config import load_config

STRATEGY = json.load(open("results/short/strategy_01_20261023_150642.json"))
rules = [
    {"rule_id": r["rule_id"], "conditions": r["conditions"], "tp": r["tp"],
     "sl": r["sl"], "capital_pct": r["capital_pct"]}
    for r in STRATEGY["strategy"]["rules_set"]
]

cfg = load_config(direction=STRATEGY["direction"])
df = run(cfg, split="train", normalize=False)
features, feature_cols, labels, meta = assemble(df)
from src.core.discretizer import FeatureDiscretizer
d = FeatureDiscretizer(); feature_info, _ = d.fit(features)
known = set(features) | {"symbol"}
dropped, kept_rules = [], []
for r in rules:
    feats_in_rule = {c.split()[0].strip("[]") for c in r["conditions"]}
    if feats_in_rule <= known:
        kept_rules.append(r)
    else:
        dropped.append(tuple(r["conditions"]))
rules = kept_rules
if dropped:
    print(f"NOTE: {len(dropped)} rules reference features absent from the "
          f"current feature regime — dropped for comparison.\n",
          dropped)
sim = CapitalManagedTradeSimulator(d.transform(features), feature_cols, labels,
                                   meta, cfg.direction,
                                   np.array([i["dont_care"] for i in feature_info]))

t0 = time.time()
m = sim.simulate_rule_set(rules)
elapsed = time.time() - t0
pnls = m["trade_pnls"]
notionals = m["trade_notional"]
equity = m["equity_curve"]
rule_idx = m["trade_rule_idx"]

# ---------- old objective reproductions
def rexp_old():
    if len(pnls) == 0:
        return -1e6  # old default F = -1e6 on an INVALID zero-trade genome too
    return float(np.mean(pnls) - 0.1 * np.std(pnls))

def rwp_old():
    if len(pnls) < 30:
        return -1e6
    cs = np.concatenate([[0], np.cumsum(pnls)])
    win = np.array([cs[i + 30] - cs[i] for i in range(len(pnls) - 29)])
    return float(0.5 * win.mean() + 0.3 * np.percentile(win, 10) - 0.2 * win.std())

def div_old():
    feat, tps, sls = {}, set(), set()
    for r in rules:
        tps.add(round(r["tp"], 1)); sls.add(round(r["sl"], 1))
        for fid in r.get("feature_ids", ()):
            feat[fid] = feat.get(fid, 0) + 1
    return float(len(feat) + 0.5 * len(tps) + 0.5 * len(sls)
                 - 2.0 * sum(c - 1 for c in feat.values() if c > 1))

# ---------- new objectives
def rexp_new():
    if len(pnls) == 0:
        return BAD
    if not np.all(np.asarray(notionals, float) > 1e-9):
        return BAD
    return float(np.mean(np.asarray(pnls, float) / np.asarray(notionals, float)) * 100)

def rwp_new():
    if len(equity) == 0 or m["executed_trades"] == 0:
        return BAD
    rp = (float(equity[-1]) / sim.initial_capital - 1.0) * 100.0
    return float(rp - 0.5 * _ulcer_pct(np.concatenate([[sim.initial_capital], equity])))

# ROB new: quick implementation with 4 folds (engine folds live on the problem;
# here we reuse the core numerics directly for the report)
def rob_new(n_folds=4):
    n = len(np.asarray(list(features.values())[0]))
    rets = []
    for i in range(n_folds):
        s = i * (n // n_folds)
        e = n if i == n_folds - 1 else (i + 1) * (n // n_folds)
        fs = sim.slice(np.arange(s, e))
        fm = fs.simulate_rule_set(rules)
        rets.append(float(fm["total_return_pct"]) if fm["executed_trades"] > 0 else 0.0)
    return float(min(rets))

per_rule = np.bincount(rule_idx.astype(int), minlength=len(rules))
print(f"strategy: {STRATEGY['strategy']['rules_set'] and len(rules) or 0} rules, "
      f"{m['executed_trades']} executed trades, final equity "
      f"{equity[-1] if len(equity) else float('nan'):,.2f}")
print("per-rule executed trades:", per_rule.tolist())
print(f"expiration time of one full evaluation: {elapsed*1000:.1f} ms "
      f"(incl. {4} fold sims)\n")
print("OLD (USD, exploitable):")
print(f"  REXP_old = {rexp_old():8.3f}   RWP_old(ovl windows) = {rwp_old():8.3f}   DIV_old = {div_old():.2f}")
print("NEW (normalized, portfolio-based):")
print(f"  REXP     = {rexp_new():8.3f} %/trade")
print(f"  RWP      = {rwp_new():8.3f} %  (Rp={ (equity[-1]/sim.initial_capital-1)*100:.2f}%  ulcer={_ulcer_pct(np.concatenate([[sim.initial_capital], equity])):.2f})")
print(f"  ROB      = {rob_new():8.3f} %  (worst of 4 chronological folds)")
