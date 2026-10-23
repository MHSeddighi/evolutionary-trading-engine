"""Tiny deterministic synthetic market for fitness/backtest correctness tests.

One rolled price series, one feature that fires "buy" on a scheduled
deterministic pattern, labels computed to honour the simulator contract
(entry = next open; TP/SL evaluated against horizon high/low).
"""
import numpy as np

from src.core.backtest import CapitalManagedTradeSimulator

N = 1200


def make_dataset(seed: int = 7, all_positive: bool = True, sparse: bool = False):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.15, N))
    op = np.concatenate([[100.0], close[:-1]])
    hor = 32
    hi, lo = np.empty(N), np.empty(N)
    for i in range(N):
        seg = close[i:i + hor]
        pad = seg[-1] if len(seg) < hor else seg
        hi[i] = max(op[i] + 0.01, pad.max())
        lo[i] = min(op[i] - 0.01, pad.min())
    hi[:hor] = np.maximum(hi[:hor], op[:hor])  # always above entry open
    lo[:hor] = np.minimum(lo[:hor], op[:hor])
    labels = {
        "label_open_next": op,
        "label_max_288": hi,
        "label_min_288": lo,
        "label_close_288": close,
        "label_max_before_min": np.ones(N, dtype=np.int8),
    }
    if not all_positive:
        codes = np.zeros(N, dtype=np.int32)
    elif sparse:
        codes = np.where(np.arange(N) % 250 == 7, 4, 0).astype(np.int32)
    else:
        codes = (np.arange(N) % 2 * 4).astype(np.int32)
    features = {"f_buy": codes, "symbol": np.zeros(N, dtype=np.int32)}
    meta = {
        "_symbol_bar_index": np.arange(N),
        "symbol_names": np.array(["TESTUSDT"]),
        "datetime": np.arange(N),
    }
    return features, ["f_buy"], labels, meta


def make_problem(all_positive=True, rwdd_penalty=0.5, n_folds=4, min_trades=10,
                 min_trades_per_rule=0, max_rules=2, min_rules=1, sparse=False):
    from src.core.fitness import MultiObjectiveProblem

    features, feature_cols, labels, meta = make_dataset(all_positive=all_positive, sparse=sparse)
    rules = [
        {"conditions": ["f_buy IS Very High"], "tp": 1.0, "sl": 0.5, "feature_ids": (0,)},
        {"conditions": ["f_buy IS Medium"], "tp": 1.2, "sl": 0.6, "feature_ids": (0,)},
    ][: max(1, max_rules)]

    class _Dec:
        this_max_rules = max_rules
        this_min_rules = min_rules

        def __init__(self):
            self.max_rules = max_rules
            self.min_rules = min_rules
            self.genes_per_slot = 5
            self.n_var = 5 * max_rules

        def decode(self, chrom):
            out = []
            for i in range(max_rules):
                if int(chrom[i * 5]) == 1:
                    r = dict(rules[min(int(chrom[i * 5 + 1]), len(rules) - 1)])
                    r.update(rule_id=int(chrom[i * 5 + 1]), capital_pct=2.0)
                    out.append(r)
            out.sort(key=lambda r: r["rule_id"])
            return {"direction": "long", "rules_set": out if len(out) >= self.min_rules else []}

        def get_bounds(self):
            xl = np.zeros(self.n_var, dtype=int)
            xu = np.zeros(self.n_var, dtype=int)
            for i in range(max_rules):
                xu[i * 5] = 1
                xu[i * 5 + 1] = 1
                xu[i * 5 + 2:] = 3
            return xl, xu

    return MultiObjectiveProblem(
        features=features, feature_cols=feature_cols, labels=labels, meta=meta,
        decoder=_Dec(), dont_cares=np.zeros(2, dtype=bool), min_trades=min_trades,
        direction="long", min_trades_per_rule=min_trades_per_rule, n_folds=n_folds,
        max_open_exposure=400, max_open_positions=40, rwdd_penalty=rwdd_penalty,
    )


def sim(all_positive=True):
    features, feature_cols, labels, meta = make_dataset(all_positive)
    return (CapitalManagedTradeSimulator(features, feature_cols, labels, meta, "long",
                                         np.zeros(2, dtype=bool)),
            features, feature_cols, labels, meta)
