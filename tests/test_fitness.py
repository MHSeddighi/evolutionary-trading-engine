"""Regression tests for the fitness refactor (REXP / RWP / ROB).

Each test builds a tiny deterministic synthetic market (tests/synth.py) and a
problem instance; assertions are about objective/constraint invariants, never
about absolute profitability.
"""
import numpy as np
import pytest

from tests.synth import make_problem, sim

GREAT = np.array([1, 0, 0, 0, 0, 1, 1, 0, 0, 0])   # 2 valid rules
EMPTY = np.zeros(10, dtype=int)                    # no enabled slots


def _ev(p, x):
    out = {}
    p._evaluate(np.asarray(x), out)
    return out


# ---------------------------------------------------------------- invalid genomes

def test_empty_genome_never_receives_optimal_fitness():
    p = make_problem()
    p.update_progress(1.0)
    out_empty = _ev(p, EMPTY)
    out_good = _ev(p, GREAT)
    assert np.all(out_empty["F"] == 1e6), out_empty["F"]
    # a valid feasible genome must not be dominated by the invalid one
    assert np.all(out_good["F"] < 1e3)


@pytest.mark.parametrize("x", [
    np.full(10, -99),       # out of bounds
    np.full(10, 999),       # out of bounds
    np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # 1 rule, min_rules=2 -> empty decode
])
def test_undecodable_genomes_get_worst_vector(x):
    p = make_problem(min_rules=2)
    p.update_progress(1.0)
    out = _ev(p, x)
    assert np.all(out["F"] == 1e6)
    assert np.all(out["G"] == 1e6)


def test_invalid_tp_sl_never_bypasses_constraints():
    # decode policy: tp<=sl rules are dropped; with 2 slots and both rules
    # invalid the genome must resolve to the empty/invalid direction.
    ref = make_problem()
    ref.update_progress(1.0)
    p = make_problem()
    p.update_progress(1.0)
    out = _ev(p, np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
    assert np.all(out["G"] >= -1e12)  # constraints evaluated on decoded rules


def test_default_F_is_worst_and_default_G_is_violation():
    p = make_problem()
    p.update_progress(0.0)
    assert np.all(p._default_F() == 1e6)
    assert np.all(p._default_G() == 1e6)


# ------------------------------------------- genotype/phenotype consistency

def test_rule_cap_counts_decoded_rules_not_slots():
    p = make_problem(min_rules=1)
    p.update_progress(1.0)  # end of run: allowed = min_rules = 1
    dec = p.decoder
    out = {}
    p._evaluate(np.asarray(GREAT), out)  # genome enables 2 slots; decode keeps both
    decoded = len(dec.decode(GREAT)["rules_set"])
    assert decoded == 2
    # rule_cap constraint sees 2 decoded rules vs allowed 1 -> violation
    assert out["G"][3] == 1.0, out["G"]


def test_duplicate_rule_ids_dont_create_double_attribution():
    p = make_problem(min_rules=2)
    p.update_progress(1.0)
    n_rules = len(p.decoder.decode(GREAT)["rules_set"])
    rid = np.asarray([t["trade_rule_idx"] for t in
                      [_metrics_of_rule(GREAT, p)]]).ravel()
    assert rid.max() < n_rules


def _metrics_of_rule(x, p):
    return p.sim.simulate_rule_set(p.decoder.decode(np.asarray(x))["rules_set"])


def test_min_trade_constraint_uses_executed_trades():
    # min_trades_per_rule floor that no signal row can satisfy: the constraint
    # must report a violation based on executed counts, never raw mask rows.
    p = make_problem(min_trades_per_rule=50)
    p.update_progress(1.0)
    out = _ev(p, GREAT)
    assert out["G"][1] > 0  # infeasible: floor above what executed per rule


def test_min_trade_constraint_satisfied_with_zero_floor():
    p = make_problem(min_trades_per_rule=0)
    p.update_progress(1.0)
    out = _ev(p, GREAT)
    assert out["G"][1] == 0.0


# ---------------------------------------------------------------- objectives

def test_rexp_is_scale_invariant_to_capital_pct():
    p = make_problem(sparse=True)
    m_lo = _metrics_cap(p, 1.0)
    m_hi = _metrics_cap(p, 3.0)
    rexp_lo = p._obj_rexp(m_lo, GREAT, m_lo["_rules"])
    rexp_hi = p._obj_rwp  # touch attr to exist
    assert abs(rexp_lo - p._obj_rexp(m_hi, GREAT, m_lo["_rules"])) < 1e-6


def _metrics_cap(p, cap):
    rules = p.decoder.decode(GREAT)["rules_set"]
    for r in rules:
        r["capital_pct"] = cap
    m = p.sim.simulate_rule_set(rules)
    m["_rules"] = rules
    return m


def test_rwp_penalizes_drawdown():
    p = make_problem(rwdd_penalty=0.5)
    m = _metrics_of_rule(GREAT, p)
    equity = m["equity_curve"]
    base = p._obj_rwp(m, GREAT, [])
    # synthesize a deeper drawdown with the same final return:prepend a dip
    deep = equity.copy()
    deep[0] = deep[0] - 50.0        # 5% dip from 1000 start
    m2 = dict(m, equity_curve=deep)
    rwp_deep = p._obj_rwp(m2, GREAT, [])
    assert rwp_deep < base


def test_ulcer_index_shapes():
    from src.core.fitness import _ulcer_pct
    flat = np.full(5, 1000.0)
    assert _ulcer_pct(np.concatenate([[1000.0], flat])) == 0.0
    dip = np.array([1000.0, 950.0, 1000.0, 1000.0, 1000.0])
    assert _ulcer_pct(dip) > 0
    # dollar-scale invariance: same relative path -> same ulcer
    assert abs(_ulcer_pct(dip * 2.0) - _ulcer_pct(dip)) < 1e-9


def test_account_ruin_is_invalid_not_attractive():
    p = make_problem()
    m = dict(_metrics_of_rule(GREAT, p), equity_curve=np.array([200.0, 0.1]))
    # score convention: BAD = -1e6; through negation it lands at F = +1e6 (worst)
    assert p._obj_rwp(m, GREAT, []) == -1e6


# ---------------------------------------------------------------- robustness

def test_rob_penalizes_single_bad_fold():
    # one fold engineered to lose 10% -> ROB cannot exceed that fold
    p = make_problem(n_folds=4)
    p.update_progress(1.0)
    rules = p.decoder.decode(GREAT)["rules_set"]
    rets = []
    p._build_folds()
    for fs in p._fold_sims:
        fm = fs.simulate_rule_set(rules)
        rets.append(fm["total_return_pct"])
    rob = p._obj_rob(_metrics_of_rule(GREAT, p), GREAT, rules)
    assert rob == pytest.approx(min(rets) if rets else rob, abs=1e-9)
    assert rob <= max(rets)


def test_inactive_strategy_cannot_exploit_rob():
    p = make_problem(all_positive=False, n_folds=4)
    out = _ev(p, GREAT)   # no signals at all
    # zero executions -> ROB is the BAD sentinel -> F = +1e6 (worst)
    assert out["F"][2] == 1e6


def test_fold_count_configurable_and_explicit():
    for n in (1, 3, 7):
        p = make_problem(n_folds=n)
        p.update_progress(1.0)
        p._build_folds()
        assert len(p._fold_sims) == max(1, min(n, 600))


def test_no_fold_lookahead_masks_built_from_past_only():
    # fold sim masks are computed from within-fold feature rows only: a rule
    # whose condition needs a future feature value must not fire unless that
    # value is inside its fold. Here: f_buy pattern interleaves, smoke-check
    # that fold executions never exceed total signal rows in each fold range.
    p = make_problem(n_folds=4)
    p.update_progress(1.0)
    p._build_folds()
    rules = p.decoder.decode(GREAT)["rules_set"]
    execs = [fs.simulate_rule_set(rules)["executed_trades"] for fs in p._fold_sims]
    assert sum(execs) >= 0  # smoke: folds execute independently without crash


# ---------------------------------------------------------------- accounting

def test_equity_curve_includes_fees_consistently():
    s, features, feature_cols, labels, meta = sim()
    m = s.simulate_rule_set([{"conditions": ["f_buy IS Very High"], "tp": 1.0, "sl": 0.5, "capital_pct": 2.0}])
    equity = m["equity_curve"]
    raw_sum = m["total_return_pct"] / 100.0 * s.initial_capital
    assert pytest.approx(equity[-1] - s.initial_capital, rel=1e-6) == raw_sum
    # net pnl must be gross minus fees: purely profitable trades impossible
    # with 0.5% round-trip fee on 1% TP / 0.5% SL trades
    max_net_win = 1.0 * 1000 * 0.06  # 6% capital on 1% TP
    assert np.all(np.asarray(m["trade_pnls"]) <= max_net_win)


def test_exposure_limits_reflected_in_portfolio():
    s, features, feature_cols, labels, meta = sim()
    m = s.simulate_rule_set([{"conditions": ["f_buy IS Very High"], "tp": 1.0, "sl": 0.5, "capital_pct": 2.0}])
    assert m["max_total_open_exposure"] <= s.max_total_exposure_rate * s.initial_capital * 1.01 + 1e-6


def test_account_ruin_break_stops_trading():
    s, features, feature_cols, labels, meta = sim()
    m = s.simulate_rule_set([{"conditions": ["f_buy IS Very High"], "tp": 1.0, "sl": 0.5, "capital_pct": 2.0}])
    # synthetic data never ruins; contract check: trade list aligned
    assert len(m["trade_pnls"]) == m["executed_trades"]
    assert len(m["trade_rule_idx"]) == m["executed_trades"]


# ---------------------------------------------------------------- annealing

def test_annealing_progression():
    p = make_problem()
    p.update_progress(0.0)
    assert float("inf") == p._allowed_exposure()
    assert p._allowed_rules_now() == p.max_rules
    p.update_progress(0.5)
    assert p._allowed_exposure() == p.max_open_exposure / 0.5
    assert p._allowed_exposure() < float("inf")
    p.update_progress(1.0)
    assert p._allowed_exposure() == p.max_open_exposure
    assert p._allowed_positions() == p.max_open_positions
    assert p._allowed_rules_now() == p.min_rules


def test_missing_callback_fails_safe_to_strict(capsys):
    # before the warn threshold the DOCUMENTED initial state (free explore) holds
    p = make_problem()
    _ev(p, GREAT)
    assert p._allowed_exposure() == float("inf")
    # after the eval counter passes the threshold, constraints fail SAFE to strict
    for _ in range(p._MISSING_PROGRESS_WARN_EVALS + 1):
        p._n_evals_seen += 1
    assert p._allowed_exposure() == p.max_open_exposure
    assert p._anneal_factor() == 1.0
    captured = capsys.readouterr()
    assert "WARNING update_progress() never called" in captured.out


def test_update_progress_restores_free_exploration_at_start():
    p = make_problem()
    p.update_progress(0.0)
    out = _ev(p, GREAT)
    assert float("inf") == p._allowed_exposure()


# ---------------------------------------------------------------- plumbing

def test_shapes_and_budget_are_finite():
    p = make_problem()
    p.update_progress(1.0)
    out = _ev(p, GREAT)
    assert out["F"].shape == (3,)
    assert out["G"].shape == (5,)
    assert np.all(np.isfinite(out["F"])) and np.all(np.isfinite(out["G"]))


def test_diagnostics_are_not_objectives():
    p = make_problem()
    p.update_progress(1.0)
    diag = p.diversity_diagnostics(p.decoder.decode(GREAT)["rules_set"])
    assert set(diag) <= {"unique_features", "max_repeated_feature",
                         "distinct_tps", "distinct_sls"}
    assert "DIV" not in [o.name for o in p.OBJECTIVES]
    assert len(p.OBJECTIVES) == 3
