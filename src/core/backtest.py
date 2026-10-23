import numpy as np
from numba import njit
from typing import Dict, List, Tuple, Optional
import pandas as pd
from src.core.discretizer import parse_condition, normalize_direction, safe_profit_factor

# Realistic execution costs (Wallex taker fees + slippage), per side.
# Wallex charges a flat 0.2% on crypto pairs (per https://wallex.ir/commission).
# Round-trip cost charged per trade = 2 * (fee_per_side + slippage_per_side).
FEE_PER_SIDE_PCT = 0.2
SLIPPAGE_PER_SIDE_PCT = 0.05
INITIAL_CAPITAL = 1000.0
LEVERAGE = 1.0
MAX_HOLD_CANDLES = 288
MAX_TOTAL_EXPOSURE_PCT = 100.0
MIN_POSITION_NOTIONAL = 1.0


def apply_dynamic_rule(feature_dict, rule_string):
    """Evaluate one condition against DISCRETIZED feature codes.

    Discretizer modes -> codes:
      binary {Inactive:0, Active:1}
      ternary {Negative:-1, Neutral:0, Positive:1}
      positive/sparse_positive quintiles {Very Low:0, Low:1, Medium:2, High:3, Very High:4}
      sparse_signed {Strong Negative:0, Weak Negative:1, Exactly Zero:2, Weak Positive:3, Strong Positive:4}
      signed (deciles) {Extreme Bearish:0 ... Extreme Bullish:9}
    """
    feature_name, value_name = parse_condition(rule_string)
    if feature_name not in feature_dict:
        raise ValueError(f"Unknown feature: {feature_name!r}")
    s = feature_dict[feature_name]
    if feature_name == "symbol":
        try:
            return s == int(value_name)
        except:
            pass
    if value_name == "Active (1)": return s == 1
    if value_name == "Inactive (0)": return s == 0
    if value_name in ["Positive", "Positive (1)"]: return s == 1
    if value_name in ["Neutral", "Neutral (0)"]: return s == 0
    if value_name in ["Negative", "Negative (-1)"]: return s == -1
    if value_name == "Very Low": return s == 0
    if value_name == "Low": return s == 1
    if value_name == "Medium": return s == 2
    if value_name == "High": return s == 3
    if value_name == "Very High": return s == 4
    if value_name == "Strong Negative": return s == 0
    if value_name == "Weak Negative": return s == 1
    if value_name in ["Exactly Zero", "Exactly Zero (No Gap)"]: return s == 2
    if value_name == "Weak Positive": return s == 3
    if value_name in ["Strong Positive", "Strong Positive (e.g. Big Up Gap)"]: return s == 4
    if value_name == "Extreme Bearish": return s == 0
    if value_name == "Strong Bearish": return s == 1
    if value_name == "Bearish": return s == 2
    if value_name == "Weak Bearish": return s == 3
    if value_name == "Neutral Negative": return s == 4
    if value_name == "Neutral Positive": return s == 5
    if value_name == "Weak Bullish": return s == 6
    if value_name == "Bullish": return s == 7
    if value_name == "Strong Bullish": return s == 8
    if value_name == "Extreme Bullish": return s == 9
    raise ValueError(f"Rule value {value_name!r} not recognised for feature {feature_name!r}.")


def build_rule_signal_mask(feature_dict, conditions):
    signal = np.ones(len(next(iter(feature_dict.values()))), dtype=bool)
    for condition in conditions:
        signal &= apply_dynamic_rule(feature_dict, condition)
    return signal


def build_rule_set_entries(data_matrix, rule_set, dont_cares, feature_dict):
    """Return (entry_idx, tp, sl, capital_pct, rule_idx) sorted chronologically.

    rule_idx = position of the owning rule in `rule_set`. Attribution semantics:
    when several rules signal at the same timestamp, the FIRST rule (in list
    order) that covers the row claims the entry via its tp/sl/capital; the
    other rules' signals at that row are rejected (they never execute).
    """
    if not rule_set or not any(r.get("signal_mask") is not None for r in rule_set):
        empty32 = np.empty(0, dtype=np.int32)
        emptyf = np.empty(0, dtype=np.float32)
        return empty32, emptyf, emptyf, emptyf, empty32.copy()

    n_rows = data_matrix.shape[0]
    assigned = np.zeros(n_rows, dtype=bool)
    idx_buf = np.empty(n_rows, dtype=np.int32)
    tp_buf = np.empty(n_rows, dtype=np.float32)
    sl_buf = np.empty(n_rows, dtype=np.float32)
    cap_buf = np.empty(n_rows, dtype=np.float32)
    rule_buf = np.zeros(n_rows, dtype=np.int32)
    count = 0

    for r_idx, r in enumerate(rule_set):
        sig = r.get("signal_mask")
        if sig is None: continue
        tp, sl, cp = float(r["tp"]), float(r["sl"]), float(r["capital_pct"])
        sig = np.asarray(sig, dtype=bool)
        match = sig & ~assigned
        idxs = np.flatnonzero(match)
        if not len(idxs): continue
        assigned[idxs] = True
        n = len(idxs)
        idx_buf[count:count + n] = idxs
        tp_buf[count:count + n] = tp
        sl_buf[count:count + n] = sl
        cap_buf[count:count + n] = cp
        rule_buf[count:count + n] = r_idx
        count += n

    if count == 0:
        empty32 = np.empty(0, dtype=np.int32)
        emptyf = np.empty(0, dtype=np.float32)
        return empty32, emptyf, emptyf, emptyf, empty32.copy()

    idx_buf = idx_buf[:count]
    order = np.argsort(idx_buf, kind='mergesort')
    return idx_buf[order], tp_buf[:count][order], sl_buf[:count][order], cap_buf[:count][order], rule_buf[:count][order]


@njit(cache=True, fastmath=True)
def _build_trade_outcomes_numba(idx_arr, tp_arr, sl_arr, max_ret, min_ret, close_ret, max_before_min, is_long):
    """
    Calculate trade outcomes for each entry.

    For LONG positions:
    - TP hit when price goes UP by tp% (max_ret >= tp)
    - SL hit when price goes DOWN by sl% (min_ret <= -sl)
    - If both hit: max_before_min==1 means max came first = TP hit first

    For SHORT positions:
    - TP hit when price goes DOWN by tp% (min_ret <= -tp)
    - SL hit when price goes UP by sl% (max_ret >= sl)
    - If both hit: max_before_min==1 means max came first = SL hit first
    """
    n = len(idx_arr)
    if n == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int8)

    ret = np.empty(n, dtype=np.float32)
    reason = np.empty(n, dtype=np.int8)  # 0=TP, 1=SL, 2=TIME

    for i in range(n):
        idx = idx_arr[i]
        tp_i, sl_i = tp_arr[i], sl_arr[i]
        s_max, s_min, s_close = max_ret[idx], min_ret[idx], close_ret[idx]
        mbm = max_before_min[idx]

        if is_long:
            # LONG: TP on upside, SL on downside
            hit_tp = s_max >= tp_i
            hit_sl = s_min <= -sl_i

            if hit_tp and hit_sl:
                # Both hit: mbm==1 means max (TP) came first
                if mbm == 1:
                    ret[i], reason[i] = tp_i, 0  # TP hit first
                else:
                    ret[i], reason[i] = -sl_i, 1  # SL hit first
            elif hit_tp:
                ret[i], reason[i] = tp_i, 0
            elif hit_sl:
                ret[i], reason[i] = -sl_i, 1
            else:
                ret[i], reason[i] = s_close, 2
        else:
            # SHORT: TP on downside, SL on upside
            hit_tp = s_min <= -tp_i  # Price moved down enough
            hit_sl = s_max >= sl_i  # Price moved up enough

            if hit_tp and hit_sl:
                # Both hit: mbm==1 means max (SL for shorts) came first
                if mbm == 1:
                    ret[i], reason[i] = -sl_i, 1  # SL hit first (loss)
                else:
                    ret[i], reason[i] = tp_i, 0  # TP hit first (profit)
            elif hit_tp:
                ret[i], reason[i] = tp_i, 0
            elif hit_sl:
                ret[i], reason[i] = -sl_i, 1
            else:
                # Time exit: for shorts, profit when price goes down (negative s_close)
                ret[i], reason[i] = -s_close, 2

    return ret, reason


@njit(cache=True, fastmath=True)
def _simulate_core_with_details(idx_arr, ret_arr, cap_pct_arr, release_idx_arr, symbol_idx_arr,
                                rule_idx_arr,
                                initial_capital, leverage, max_total_exposure_rate,
                                min_position_notional, round_trip_fee_rate):
    """
    Core simulation loop with portfolio exposure management.

    Processes trades in chronological order, respecting:
    - Maximum total exposure limits
    - Position holding periods (release indices)
    - Capital allocation per trade
    - Minimum position size requirements
    """
    n = len(idx_arr)

    # Open positions tracking arrays
    open_rel = np.empty(n, dtype=np.int32)  # Release indices for open positions
    open_not = np.empty(n, dtype=np.float32)  # Notional values of open positions
    open_pnl = np.empty(n, dtype=np.float32)  # P&L of open positions
    open_symbol = np.empty(n, dtype=np.int32)  # Symbol codes for open positions
    open_entry = np.empty(n, dtype=np.int32)  # Entry indices for open positions
    open_rule = np.empty(n, dtype=np.int32)  # Owning rule slot for each open position
    n_open = 0

    # Account state
    equity = initial_capital
    peak = equity
    max_dd = 0.0
    total_exp = 0.0
    max_total_exp = 0.0  # Track maximum exposure reached

    # Symbol exposure tracking (fixed size for numba)
    max_symbols = 100
    symbol_exp = np.zeros(max_symbols, dtype=np.float32)

    # Statistics accumulators
    dd_sum = 0.0
    dd_count = 0
    executed = 0
    skipped = 0
    notional_sum = 0.0
    max_simul = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    # Trade history arrays — convention: equity_curve records equity at position
    # close events only (mark-to-market at close). Trades resolving TP/SL to a
    # horizon price can't expose intra-hold path equity, so unrealized pnl on
    # still-open positions is not reflected until their release row. Fees and
    # slippage are included in every recorded pnl (net). This is a documented,
    # deliberate convention of this simulator.
    trade_pnls = np.empty(n, dtype=np.float32)
    trade_entry_idx = np.empty(n, dtype=np.int32)
    trade_exit_idx = np.empty(n, dtype=np.int32)
    trade_notional = np.empty(n, dtype=np.float32)
    trade_rule_idx = np.empty(n, dtype=np.int32)
    equity_curve = np.empty(n, dtype=np.float32)
    equity_idx = 0

    for i in range(n):
        cur_idx = idx_arr[i]

        # Process due positions (trades that have reached their release index)
        if n_open > 0:
                due_mask = np.empty(n_open, dtype=np.bool_)
                due_cnt = 0
                for j in range(n_open):
                    if open_rel[j] <= cur_idx:
                        due_mask[j] = True
                        due_cnt += 1
                    else:
                        due_mask[j] = False

                if due_cnt > 0:
                    pnl_sum = 0.0
                    for j in range(n_open):
                        if due_mask[j]:
                            pnl = open_pnl[j]
                            pnl_sum += pnl

                            # Record trade outcome
                            trade_pnls[executed] = pnl
                            trade_entry_idx[executed] = open_entry[j]
                            trade_exit_idx[executed] = cur_idx
                            trade_notional[executed] = open_not[j]
                            trade_rule_idx[executed] = open_rule[j]
                            executed += 1

                            # Update statistics
                            if pnl > 0.0:
                                wins += 1
                                gross_profit += pnl
                            elif pnl < 0.0:
                                losses += 1
                                gross_loss += -pnl

                            # Release exposure
                            total_exp -= open_not[j]
                            sym_idx = open_symbol[j]
                            symbol_exp[sym_idx] -= open_not[j]

                    # Update equity and drawdown
                    equity += pnl_sum
                    equity_curve[equity_idx] = equity
                    equity_idx += 1

                    if equity > peak:
                        peak = equity

                    if peak > 0.0:
                        dd = (peak - equity) / peak * 100.0
                        if dd > max_dd:
                            max_dd = dd
                        dd_sum += dd
                        dd_count += 1

                    # Remove closed positions
                    w = 0
                    for j in range(n_open):
                        if not due_mask[j]:
                            open_rel[w] = open_rel[j]
                            open_not[w] = open_not[j]
                            open_pnl[w] = open_pnl[j]
                            open_symbol[w] = open_symbol[j]
                            open_entry[w] = open_entry[j]
                            open_rule[w] = open_rule[j]
                            w += 1
                    n_open = w

        # Check account ruin
        if equity <= 0.0:
            break

        # Calculate position size respecting exposure limits
        target = equity * (cap_pct_arr[i] / 100.0) * leverage
        max_total = equity * max_total_exposure_rate * leverage
        remaining = max(0.0, max_total - total_exp)

        sym_idx = symbol_idx_arr[i]
        remaining_per_symbol = max(0.0, max_total - symbol_exp[sym_idx])
        remaining = min(remaining, remaining_per_symbol)

        pos_notional = min(target, remaining)

        # Skip if position is too small
        if pos_notional < min_position_notional:
            skipped += 1
            continue

        # Calculate P&L for this trade
        ret_rate = ret_arr[i] / 100.0
        gross_pnl = pos_notional * ret_rate
        fee = pos_notional * round_trip_fee_rate
        net_pnl = gross_pnl - fee

        # Open new position
        open_rel[n_open] = release_idx_arr[i]
        open_not[n_open] = pos_notional
        open_pnl[n_open] = net_pnl
        open_symbol[n_open] = sym_idx
        open_entry[n_open] = cur_idx
        open_rule[n_open] = rule_idx_arr[i]
        n_open += 1

        # Update exposure
        total_exp += pos_notional
        symbol_exp[sym_idx] += pos_notional

        # Track maximums
        if total_exp > max_total_exp:
            max_total_exp = total_exp

        notional_sum += pos_notional
        if n_open > max_simul:
            max_simul = n_open

    # Close any remaining open positions at end of data
    if n_open > 0:
        final_pnl = 0.0
        for j in range(n_open):
            pnl = open_pnl[j]
            final_pnl += pnl

            trade_pnls[executed] = pnl
            trade_entry_idx[executed] = open_entry[j]
            trade_exit_idx[executed] = n - 1  # Use last index
            trade_notional[executed] = open_not[j]
            trade_rule_idx[executed] = open_rule[j]
            executed += 1

            if pnl > 0.0:
                wins += 1
                gross_profit += pnl
            elif pnl < 0.0:
                losses += 1
                gross_loss += -pnl

        equity += final_pnl
        equity_curve[equity_idx] = equity
        equity_idx += 1

        if equity > peak:
            peak = equity

        if peak > 0.0:
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
            dd_sum += dd
            dd_count += 1

    # Calculate final metrics
    avg_dd = dd_sum / max(dd_count, 1)
    total_ret = (equity / initial_capital - 1.0) * 100.0

    return (total_ret, max_dd, avg_dd, executed, wins, losses,
            gross_profit, gross_loss, notional_sum, skipped,
            max_simul, max_total_exp,  # Now returns actual max exposure
            trade_pnls[:executed], trade_entry_idx[:executed],
            trade_exit_idx[:executed], trade_notional[:executed],
            trade_rule_idx[:executed],
            equity_curve[:equity_idx])


class CapitalManagedTradeSimulator:
    """
    Capital-Managed Trade Simulator with Portfolio Exposure Control.

    Handles both long and short trades with:
    - Dynamic position sizing based on capital allocation
    - Maximum total exposure limits
    - Per-symbol exposure tracking
    - Trade holding period management
    - TP/SL order resolution (which hits first)
    """

    def __init__(self, feature_dict, feature_cols, labels, meta, trade_direction, dont_cares,
                 fee_per_side_pct=FEE_PER_SIDE_PCT, slippage_per_side_pct=SLIPPAGE_PER_SIDE_PCT,
                 initial_capital=INITIAL_CAPITAL, leverage=LEVERAGE,
                 max_hold_candles=MAX_HOLD_CANDLES, max_total_exposure_pct=MAX_TOTAL_EXPOSURE_PCT,
                 min_position_notional=MIN_POSITION_NOTIONAL):

        self.data_matrix = np.column_stack([feature_dict[c] for c in feature_cols])
        self.dont_cares = dont_cares
        self.feature_dict = feature_dict
        self.feature_cols = feature_cols
        self.labels = labels

        # Calculate forward returns
        entry = labels["label_open_next"].astype(np.float32, copy=False)
        self.max_ret = ((labels["label_max_288"].astype(np.float32, copy=False) - entry) / entry * 100.0)
        self.min_ret = ((labels["label_min_288"].astype(np.float32, copy=False) - entry) / entry * 100.0)
        self.close_ret = ((labels["label_close_288"].astype(np.float32, copy=False) - entry) / entry * 100.0)
        self.max_before_min = labels["label_max_before_min"].astype(np.int8, copy=False)

        # Handle symbol identification
        if "symbol" in feature_dict:
            self.symbol_codes = feature_dict["symbol"].astype(np.int32)
        elif "symbol" in meta and meta["symbol"] is not None:
            symbol_strings = np.asarray(meta["symbol"], dtype=str)
            self.symbol_codes, _ = pd.factorize(symbol_strings)
            self.feature_dict["symbol"] = self.symbol_codes
        else:
            self.symbol_codes = np.zeros(len(self.data_matrix), dtype=np.int32)

        self.symbol_bar_index = np.asarray(meta.get("_symbol_bar_index", np.arange(len(self.data_matrix))),
                                           dtype=np.int32)

        # Trading parameters
        self.is_long = normalize_direction(trade_direction) == "long"
        self.fee_per_side_pct = fee_per_side_pct
        self.slippage_per_side_pct = slippage_per_side_pct
        # full round-trip friction: taker fee + adverse slippage on BOTH sides
        self.round_trip_fee_rate = (fee_per_side_pct + slippage_per_side_pct) * 2.0 / 100.0
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.max_total_exposure_rate = max_total_exposure_pct / 100.0
        self.min_position_notional = min_position_notional
        self.release_index = self._precompute_release_indices(max_hold_candles)

    def _precompute_release_indices(self, max_hold):
        """
        Precompute the release row index for every possible entry row.
        Each trade reserves exposure until entry_bar + max_hold candles.
        """
        n = len(self.symbol_codes)
        release = np.full(n, n, dtype=np.int32)

        for sym_code in np.unique(self.symbol_codes):
            mask = self.symbol_codes == sym_code
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue

            bars = self.symbol_bar_index[idxs]
            order = np.argsort(bars)
            sorted_idxs = idxs[order]
            sorted_bars = bars[order]

            target_bars = sorted_bars + max_hold
            pos = np.searchsorted(sorted_bars, target_bars, side="left")
            valid = pos < len(sorted_idxs)
            release[sorted_idxs[valid]] = sorted_idxs[pos[valid]]

        return release

    def _precompute_masks(self, rule_set):
        """Precompute signal masks for each rule in the rule set."""
        result = []
        for rule in rule_set:
            r = dict(rule)
            if "signal_mask" not in r and "conditions" in r:
                valid = [c for c in r["conditions"] if "None" not in c and " IS " in c]
                if valid:
                    mask = np.ones(len(self.max_ret), dtype=bool)
                    for cond in valid:
                        mask &= apply_dynamic_rule(self.feature_dict, cond)
                    r["signal_mask"] = mask
            result.append(r)
        return result

    def _empty_metrics(self):
        """Return empty metrics when no trades are generated."""
        return {
            "total_return_pct": 0.0,
            "max_drawdown_pct": 100.0,
            "executed_trades": 0,
            "win_rate": 0.0,
            "account_ruined": True,
            "loss_count": 0,
            "time_closed_count": 0,
            "raw_signal_count": 0,
            "final_equity": 0.0,
            "profit_factor": 0.0,
            "avg_position_notional": 0.0,
            "skipped_min_notional_count": 0,
            "max_simultaneous_positions": 0,
            "max_total_open_exposure": 0.0,
            "avg_drawdown_pct": 100.0,
            "trade_pnls": np.array([]),
            "trade_entry_idx": np.array([]),
            "trade_exit_idx": np.array([]),
            "trade_notional": np.array([]),
            "trade_rule_idx": np.array([], dtype=np.int32),
            "equity_curve": np.array([]),
        }

    def simulate_rule_set(self, rule_set, return_per_rule_masks=False):
        """
        Simulate a rule set with capital management and exposure control.

        Args:
            rule_set: List of rules with conditions, tp, sl, capital_pct
            return_per_rule_masks: If True, also return individual rule masks

        Returns:
            metrics dict with performance statistics
            (and optionally per-rule signal masks)
        """
        # Precompute signal masks
        rule_set = self._precompute_masks(rule_set)

        # Build trade entries (5th array: owning rule slot per entry)
        idx_arr, tp_arr, sl_arr, cap_pct_arr, rule_idx_arr = build_rule_set_entries(
            self.data_matrix, rule_set, self.dont_cares, self.feature_dict)

        if len(idx_arr) == 0:
            if return_per_rule_masks:
                masks = [r.get('signal_mask', np.zeros(len(self.max_ret), dtype=bool)) for r in rule_set]
                return self._empty_metrics(), masks
            return self._empty_metrics()

        # Get release indices
        release_idx_arr = self.release_index[idx_arr]

        # Calculate trade outcomes
        ret_arr, _ = _build_trade_outcomes_numba(
            idx_arr, tp_arr, sl_arr,
            self.max_ret, self.min_ret, self.close_ret,
            self.max_before_min, self.is_long
        )

        # Get symbol indices
        symbol_idx_arr = self.symbol_codes[idx_arr]

        # Run simulation
        (total_ret, max_dd, avg_dd, executed, wins, losses,
         gross_profit, gross_loss, notional_sum, skipped,
         max_simul, max_total_exp,
         trade_pnls, trade_entry_idx, trade_exit_idx, trade_notional,
         trade_rule_idx, equity_curve) = _simulate_core_with_details(
            idx_arr, ret_arr, cap_pct_arr, release_idx_arr, symbol_idx_arr,
            rule_idx_arr,
            self.initial_capital, self.leverage, self.max_total_exposure_rate,
            self.min_position_notional, self.round_trip_fee_rate)

        # Calculate derived metrics
        win_rate = (wins / executed * 100.0) if executed > 0 else 0.0
        profit_factor = safe_profit_factor(gross_profit, gross_loss)
        avg_notional = notional_sum / executed if executed > 0 else 0.0
        final_eq = self.initial_capital * (1.0 + total_ret / 100.0)

        # slice to the real count: np.empty buffers hold
        # uninitialized garbage past the executed trades, which poisons objectives
        # (robust expectancy/fold splits averaged over un-initialized rows)
        metrics = {
            "total_return_pct": total_ret,
            "max_drawdown_pct": max_dd,
            "avg_drawdown_pct": avg_dd,
            "executed_trades": executed,
            "win_rate": win_rate,
            "account_ruined": False,
            "loss_count": losses,
            "win_count": wins,
            "time_closed_count": 0,
            "raw_signal_count": len(idx_arr),
            "final_equity": final_eq,
            "profit_factor": profit_factor,
            "avg_position_notional": avg_notional,
            "skipped_min_notional_count": skipped,
            "max_simultaneous_positions": max_simul,
            "max_total_open_exposure": max_total_exp,
            "trade_pnls": np.asarray(trade_pnls).copy(),
            "trade_entry_idx": np.asarray(trade_entry_idx).copy(),
            "trade_exit_idx": np.asarray(trade_exit_idx).copy(),
            "trade_notional": np.asarray(trade_notional).copy(),
            "trade_rule_idx": np.asarray(trade_rule_idx, dtype=np.int32).copy(),
            "equity_curve": np.asarray(equity_curve).copy(),
        }

        if return_per_rule_masks:
            masks = [r.get('signal_mask', np.zeros(len(self.max_ret), dtype=bool)) for r in rule_set]
            return metrics, masks
        return metrics

    def slice(self, indices):
        """Create a new simulator with a subset of the data."""
        sliced_feature_dict = {k: v[indices] for k, v in self.feature_dict.items()}
        sliced_labels = {k: v[indices] for k, v in self.labels.items()}
        sliced_meta = {"_symbol_bar_index": self.symbol_bar_index[indices]}

        return CapitalManagedTradeSimulator(
            feature_dict=sliced_feature_dict,
            feature_cols=self.feature_cols,
            labels=sliced_labels,
            meta=sliced_meta,
            trade_direction="long" if self.is_long else "short",
            dont_cares=self.dont_cares,
            fee_per_side_pct=self.fee_per_side_pct,
            slippage_per_side_pct=self.slippage_per_side_pct,
            initial_capital=self.initial_capital,
            leverage=self.leverage,
            max_hold_candles=MAX_HOLD_CANDLES,
            max_total_exposure_pct=self.max_total_exposure_rate * 100.0,
            min_position_notional=self.min_position_notional
        )
