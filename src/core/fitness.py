"""Multi-objective fitness for the pooled NSGA-II ruleset optimizer.

Objectives (all maximized internally, expressed as F = -score for pymoo):

REXP  -- Return efficiency (capital efficiency, units: % return per trade,
         averaged on each trade's OWN notional)
         mean(trade_pnls / trade_notional) * 100.
         Exactly scale-invariant to capital_pct: pnl and notional of each
         trade scale together, so the sizing gene cannot inflate the score.
         (A portfolio-sum ratio would NOT be exactly invariant once compounding
         makes notional paths quadratic in the size gene.) Inactive/invalid ->
         BAD.

RWP   -- Risk-adjusted portfolio performance (units: % equity)
         Rp - lambda * UlcerIndex, both computed on the simulated portfolio
         equity curve (net of fees/slippage, exposure-managed).
         Rp = (equity[-1] / initial_capital - 1) * 100.
         Ulcer computed on the percentage drawdown series of that curve.
         lambda = rwdd_penalty (configurable; 1 Ulcer point costs lambda Rp
         points). Inactive/empty curve/account ruin handled explicitly.

ROB   -- Chronological robustness (units: % equity)
         Worst normalized portfolio return across n_folds chronological folds,
         each simulated independently with the SAME execution rules (equity
         reset per fold, no position carry). A fold with fewer executed trades
         than min_fold_trades scores 0.0 so inactivity cannot earn robustness.

Constraints (>= 0 feasible):
  min_rules        decoded rules grow toward min_rules over the run
  min_trades       per-decoded-rule executed-trade floor grows over the run
  max_exposure     max_total_open_exposure cap grows tighter over the run
  rule_cap         decoded-rule count cap shrinks over the run
  max_positions    max_simultaneous_positions cap grows tighter over the run

Invalid/undecodable genomes: F = +1e6 on every objective (worst possible after
negation) AND G = +1e6 on every constraint. At anneal factor 0 constraints are
satisfied even for empty genomes, so the largest positive F sentinel is what
keeps them at the bottom of every non-dominated front.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from pymoo.core.problem import Problem
from src.core.backtest import CapitalManagedTradeSimulator


BAD = -1e6  # bad SCORE on the maximize convention; F (= -score) sees +1e6 (worst)


@dataclass
class Objective:
    name: str
    display: str
    minimize: bool = True
    compute: Callable = None


@dataclass
class Constraint:
    name: str
    compute: Callable = None


def _ulcer_pct(equity):
    """Ulcer index (%) from an equity series that includes the starting point."""
    eq = np.asarray(equity, dtype=np.float64)
    peaks = np.maximum.accumulate(eq)
    dd = np.where(peaks > 0, (peaks - eq) / peaks * 100.0, 0.0)
    return float(np.sqrt(np.mean(dd * dd)))


class MultiObjectiveProblem(Problem):

    _MISSING_PROGRESS_WARN_EVALS = 2000  # after this many evals with no callback: q->1

    def __init__(self, features, feature_cols, labels, meta, decoder, dont_cares,
                 min_trades, direction, min_trades_per_rule,
                 n_folds,
                 max_open_exposure,
                 max_open_positions,
                 rwdd_penalty: float = 0.5):
        self.decoder = decoder
        self.features = features
        self.min_trades = max(int(min_trades), 1)
        self.min_trades_per_rule = int(min_trades_per_rule)
        self.max_rules = decoder.max_rules
        self.min_rules = decoder.min_rules
        self.genes_per_slot = decoder.genes_per_slot
        self.n_var = decoder.n_var
        self.generation = 0
        self.n_folds = max(int(n_folds), 1)
        self.max_open_exposure = max_open_exposure
        self.max_open_positions = max_open_positions
        self.rwdd_penalty = float(rwdd_penalty)

        # Constraint annealing: at the start of the run NO constraint is active
        # (q=0: free exploration; unlimited exposure/positions, no rule/count or
        # trade floors). Limits ramp LINEARLY on generation progress and reach
        # their real values exactly at the end of the run. Progress is fed by
        # GenUpdater every generation (wired in main.py); if the callback never
        # fires the annealing fails SAFE to the strict end-state (q=1) instead
        # of silently disabling the constraints forever.
        self._progress = 0.0
        self._progress_seen = False
        self._progress_fallback_warned = False
        self._n_evals_seen = 0

        self.sim = CapitalManagedTradeSimulator(
            features, feature_cols, labels, meta, direction, dont_cares
        )
        self._fold_sims = None
        self._fold_start_idx = None

        self.OBJECTIVES = [
            Objective("ReturnEfficiency", "REXP", False, self._obj_rexp),
            Objective("RiskAdjustedPortfolio", "RWP", False, self._obj_rwp),
            Objective("Robustness", "ROB", False, self._obj_rob),
        ]

        self.CONSTRAINTS = [
            Constraint("min_rules", self._con_min_rules),
            Constraint("min_trades", self._con_min_trades_per_rule),
            Constraint("max_exposure", self._con_max_exposure),
            Constraint("rule_cap", self._con_rule_cap),
            Constraint("max_positions", self._con_max_positions),
        ]

        xl, xu = decoder.get_bounds()
        super().__init__(
            n_var=self.n_var, n_obj=len(self.OBJECTIVES), n_constr=len(self.CONSTRAINTS),
            xl=xl, xu=xu, vtype=int, elementwise=True
        )

    # ---------------------------------------------------------------- folds

    def _build_folds(self):
        """Chronological, contiguous folds of the training rows. Each fold is an
        independently-built simulator (equity/exposure reset; positions never
        carried across fold boundaries). Signal masks are computed ONCE on the
        full sample per rule and row-sliced per fold, so the per-genome step
        only re-runs the (numba) core sims. Resolution caveat (documented): the
        TP/SL outcome labels near a fold edge were computed on the full symbol
        timeline, so outcome resolution may use prices up to the horizon past
        the fold edge — no entry SIGNAL ever sees future features."""
        if self._fold_sims is not None:
            return
        n = len(next(iter(self.features.values())))
        n_folds = min(self.n_folds, max(n // 2, 1))
        fold_size = n // n_folds
        self._fold_sims = []
        self._fold_start_idx = []
        for i in range(n_folds):
            start = i * fold_size
            end = n if i == n_folds - 1 else (i + 1) * fold_size
            if end - start < 2:
                continue
            self._fold_sims.append(self.sim.slice(np.arange(start, end)))
            self._fold_start_idx.append(start)

    # ------------------------------------------------------------- objectives

    def _obj_rexp(self, metrics, chrom, decoded_rules):
        """REXP = 100 * mean(net pnl / notional) — average per-trade net
        return rate. Exactly invariant to capital_pct."""
        pnls = metrics.get("trade_pnls", np.array([]))
        notionals = metrics.get("trade_notional", np.array([]))
        if len(pnls) == 0:
            return BAD
        notional = np.asarray(notionals, dtype=np.float64)
        if not np.all(notional > 1e-9):
            return BAD
        return float(np.mean(np.asarray(pnls, dtype=np.float64) / notional) * 100.0)

    def _obj_rwp(self, metrics, chrom, decoded_rules):
        """RWP = Rp - lambda * Ulcer. Rp in % equity, Ulcer in % drawdown.
        lambda = rwdd_penalty (default 0.5: 1 Ulcer point costs 0.5 return points).
        Ruin (equity <= 0) -> BAD. Flat-no-trade policies end with Rp = 0 and
        ulcer = 0 -> RWP = 0 (attractive in isolation) — the ROB fold guard and
        min-trade constraints ensure they cannot dominate."""
        equity = metrics.get("equity_curve", np.array([]))
        if len(equity) == 0 or metrics.get("executed_trades", 0) == 0:
            return BAD
        final = float(equity[-1])
        # ruin economics: equity at/below the min position notional can never
        # open another trade ever again — account is functionally ruined
        if not np.isfinite(final) or final <= self.sim.min_position_notional:
            return BAD
        rp = (final / self.sim.initial_capital - 1.0) * 100.0
        ul = _ulcer_pct(np.concatenate([[self.sim.initial_capital], equity]))
        return float(rp - self.rwdd_penalty * ul)

    def _obj_rob(self, metrics, chrom, decoded_rules):
        """ROB = min over folds of (fold portfolio return %), with per-fold
        executed-trade floor; folds with too few executed trades score 0.0.
        A strategy that never executes anywhere gets BAD (no free robustness)."""
        if metrics.get("executed_trades", 0) == 0:
            return BAD
        self._build_folds()
        min_fold_trades = max(2, self.min_trades // max(len(self._fold_sims), 1))
        if len(decoded_rules) == 0:
            return BAD
        fold_rets = []
        for sim in self._fold_sims:
            fm = sim.simulate_rule_set(decoded_rules)
            trades = int(fm.get("executed_trades", 0))
            if trades > 0 and np.isfinite(fm["total_return_pct"]):
                ret = float(fm["total_return_pct"])
            else:
                ret = 0.0                      # inactive/under-traded fold
            if 0 <= trades < min_fold_trades:  # inactivity buys no robustness
                ret = min(ret, 0.0)
            fold_rets.append(ret)
        return float(min(fold_rets))

    # ------------------------------------------------- non-reward diagnostics
    # (distinct-feature / redundancy counts are diagnostics only; they must
    # NEVER flow into F. Report via the metrics dict instead.)
    def diversity_diagnostics(self, decoded_rules):
        counts, tps, sls = {}, set(), set()
        for rule in decoded_rules:
            tps.add(round(rule["tp"], 1))
            sls.add(round(rule["sl"], 1))
            for fid in rule["feature_ids"]:
                counts[fid] = counts.get(fid, 0) + 1
        return {
            "unique_features": len(counts),
            "max_repeated_feature": max(counts.values()) if counts else 0,
            "distinct_tps": len(tps),
            "distinct_sls": len(sls),
        }

    # ------------------------------------------------------------ annealing

    def _anneal_factor(self):
        """0.0..1.0 — how far constraints have hardened. 0.0 = documented
        initial state (free exploration), 1.0 = end of run (all real limits).

        Progress arrives via GenUpdater after each generation; the FIRST
        sampling evaluation always runs before any callback can fire, so an
        unseen progress indicator keeps the DOCUMENTED initial state (q=0).
        A genuinely missing callback is caught by the eval counter: after
        _MISSING_PROGRESS_WARN_EVALS evaluations it fails SAFE to the strict
        end-state (q=1) with a loud one-time warning."""
        if not self._progress_seen:
            if self._n_evals_seen > self._MISSING_PROGRESS_WARN_EVALS:
                if not self._progress_fallback_warned:
                    print("[fitness] WARNING update_progress() never called after "
                          f"{self._n_evals_seen} evals; constraints forced to strict "
                          "end-state (q=1).")
                    self._progress_fallback_warned = True
                return 1.0
            return 0.0  # documented initial state
        return float(np.clip(self._progress, 0.0, 1.0))

    def _allowed_exposure(self):
        q = self._anneal_factor()
        if q <= 0.0:
            return float("inf")     # inactive at start of run
        return self.max_open_exposure / q

    def _allowed_positions(self):
        q = self._anneal_factor()
        if q <= 0.0:
            return float("inf")     # inactive at start of run
        return self.max_open_positions / q

    def _allowed_rules_now(self):
        q = self._anneal_factor()
        span = self.max_rules - self.min_rules
        return min(self.max_rules, self.min_rules + int(round((1.0 - q) * span)))

    def _con_max_exposure(self, metrics, chrom, decoded_rules):
        exposure = metrics.get("max_total_open_exposure", 0)
        limit = self._allowed_exposure()
        return 0.0 if exposure <= limit else float(exposure - limit)

    def _con_max_positions(self, metrics, chrom, decoded_rules):
        positions = metrics.get("max_simultaneous_positions", 0)
        limit = self._allowed_positions()
        return 0.0 if positions <= limit else float(positions - limit)

    def _con_rule_cap(self, metrics, chrom, decoded_rules):
        # counts the rules that SURVIVED decoding — not the raw genome slots.
        over = len(decoded_rules) - self._allowed_rules_now()
        return float(over) if over > 0 else 0.0

    def _con_min_rules(self, metrics, chrom, decoded_rules):
        q = self._anneal_factor()
        eff = int(round(self.min_rules * q))
        return float(eff - len(decoded_rules))

    def _con_min_trades_per_rule(self, metrics, chrom, decoded_rules):
        # executed-trade attribution: only trades the portfolio actually
        # executed count, attributed to the rule whose tp/sl/cap governed it.
        q = self._anneal_factor()
        eff = int(round(self.min_trades_per_rule * q))
        rule_idx = metrics.get("trade_rule_idx", np.array([]))
        if len(rule_idx) == 0 or len(decoded_rules) == 0:
            return float(eff) if eff > 0 else 0.0
        counts = np.bincount(np.asarray(rule_idx, dtype=int), minlength=len(decoded_rules))
        worst = float(counts.min())
        return max(float(eff - worst), 0.0)

    # ------------------------------------------------------------- plumbing

    def _default_F(self):
        # F = -score for maximized objectives => genuinely WORST vector.
        return np.full(len(self.OBJECTIVES), -BAD, dtype=np.float64)

    def _default_G(self):
        return np.full(len(self.CONSTRAINTS), -BAD, dtype=np.float64)

    def update_progress(self, progress: float):
        self._progress = float(np.clip(progress, 0.0, 1.0))
        self._progress_seen = True
        self.generation += 1

    def _evaluate(self, x, out, *args, **kwargs):
        self._n_evals_seen += 1
        try:
            chrom = x.astype(np.int32).ravel()
            rule_set = self.decoder.decode(chrom).get("rules_set", [])
        except Exception:
            rule_set = []

        if not rule_set:
            out["F"] = self._default_F()
            out["G"] = self._default_G()
            return

        # simulate the REAL executed portfolio once, on the decoded rules
        try:
            metrics = self.sim.simulate_rule_set(rule_set)
        except Exception:
            out["F"] = self._default_F()
            out["G"] = self._default_G()
            return

        F = np.empty(len(self.OBJECTIVES), dtype=np.float64)
        for i, obj in enumerate(self.OBJECTIVES):
            try:
                v = obj.compute(metrics, chrom, rule_set)
            except Exception:
                v = BAD
            if not np.isfinite(v):
                v = BAD
            F[i] = -v if not obj.minimize else v

        G = np.empty(len(self.CONSTRAINTS), dtype=np.float64)
        for i, con in enumerate(self.CONSTRAINTS):
            try:
                invalid = con.compute(metrics, chrom, rule_set)
            except Exception:
                invalid = -BAD
            G[i] = invalid

        # constraint convention: G > 0 = violation. NaN/inf/negated errors must
        # fall on the VIOLATING side (+1e6), never the feasible side.
        out["F"] = np.nan_to_num(F, nan=-BAD, posinf=-BAD, neginf=-BAD)
        out["G"] = np.nan_to_num(G, nan=-BAD, posinf=-BAD, neginf=0.0)
