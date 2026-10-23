from typing import Dict, List, Optional, Tuple
import numpy as np
from src.core.discretizer import normalize_direction
from src.initialization import RulePool


class Decoder:
    def __init__(
            self,
            max_rules: int,
            min_rules: int,
            direction: str,
            rule_pool: RulePool,
            tp_offset_min: float = -5.0,
            tp_offset_max: float = 1.0,
            tp_offset_scale: float = 0.5,
            sl_offset_min: float = -1.5,
            sl_offset_max: float = 0.5,
            sl_offset_scale: float = 0.3,
            cap_min: float = 1.0,
            cap_max: float = 5.0,
            cap_scale: float = 1.0,
            min_tp: float = 0.2,
            min_sl: float = 0.2,
    ):
        self.direction = normalize_direction(direction)
        self.max_rules = max_rules
        self.min_rules = min_rules
        self.rule_pool = rule_pool
        self.n_rules = len(rule_pool.rules)

        self.tp_offset_min = tp_offset_min
        self.tp_offset_max = tp_offset_max
        self.tp_offset_scale = tp_offset_scale
        self.tp_offset_steps = int((tp_offset_max - tp_offset_min) / tp_offset_scale) + 1

        self.sl_offset_min = sl_offset_min
        self.sl_offset_max = sl_offset_max
        self.sl_offset_scale = sl_offset_scale
        self.sl_offset_steps = int((sl_offset_max - sl_offset_min) / sl_offset_scale) + 1

        self.cap_min = cap_min
        self.cap_max = cap_max
        self.cap_scale = cap_scale
        self.cap_steps = int(round((cap_max - cap_min) / cap_scale)) + 1

        self.min_tp = min_tp
        self.min_sl = min_sl

        self.genes_per_slot = 5
        self.n_var = self.max_rules * self.genes_per_slot
        self.feature_to_id = {}
        self._prepare_rule_feature_ids()
        self.xl, self.xu = self._build_bounds()

    def _prepare_rule_feature_ids(self):
        next_id = 0

        for rule in self.rule_pool.rules:
            feature_ids = []

            for cond in rule["conditions"]:
                if cond.startswith("[symbol]"):
                    continue

                feature = cond.split("]", 1)[0]

                if feature not in self.feature_to_id:
                    self.feature_to_id[feature] = next_id
                    next_id += 1

                feature_ids.append(self.feature_to_id[feature])

            rule["feature_ids"] = tuple(feature_ids)

    def _build_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        xl = np.zeros(self.n_var, dtype=int)
        xu = np.zeros(self.n_var, dtype=int)

        for i in range(self.max_rules):
            base = i * self.genes_per_slot
            xl[base] = 0
            xu[base] = 1
            xl[base + 1] = 0
            xu[base + 1] = self.n_rules - 1
            xl[base + 2] = 0
            xu[base + 2] = self.tp_offset_steps - 1
            xl[base + 3] = 0
            xu[base + 3] = self.sl_offset_steps - 1
            xl[base + 4] = 0
            xu[base + 4] = self.cap_steps - 1

        return xl, xu

    def _offset_from_idx(self, idx: int, min_val: float, scale: float, steps: int) -> float:
        idx = int(np.clip(idx, 0, steps - 1))
        return round(min_val + idx * scale, 2)

    def decode(self, chrom: np.ndarray) -> Dict:
        """Genome -> ruleset.

        Documented rejection policy (silent, deliberate):
        - slot disabled   -> rule dropped
        - duplicate rid   -> later duplicate dropped (first kept)
        - tp <= sl        -> rule dropped (unexecutable TP/SL pair)
        - final count < min_rules -> empty ruleset (INVALID genome)
        The count of rules returned here is the single source of truth; the
        fitness rule-cap and min-trades constraints count THESE rules, never
        the raw enabled slots."""
        rules = []
        used = set()

        for i in range(self.max_rules):
            base = i * self.genes_per_slot

            enabled = int(chrom[base])
            if enabled == 0:
                continue

            rid = int(chrom[base + 1])
            if rid in used:
                continue
            used.add(rid)

            tp_off_idx = int(chrom[base + 2])
            sl_off_idx = int(chrom[base + 3])
            cap_idx = int(chrom[base + 4])

            pool_rule = self.rule_pool.rules[rid]
            base_tp = pool_rule.get("tp", 1.0)
            base_sl = pool_rule.get("sl", 0.5)

            tp_off = self._offset_from_idx(tp_off_idx, self.tp_offset_min, self.tp_offset_scale, self.tp_offset_steps)
            sl_off = self._offset_from_idx(sl_off_idx, self.sl_offset_min, self.sl_offset_scale, self.sl_offset_steps)

            tp = round(max(self.min_tp, base_tp + tp_off), 2)
            sl = round(max(self.min_sl, base_sl + sl_off), 2)

            if tp <= sl:
                continue

            cap = self._offset_from_idx(cap_idx, self.cap_min, self.cap_scale, self.cap_steps)

            rules.append({
                "rule_id": rid,
                "conditions": pool_rule["conditions"].copy(),
                "feature_ids": pool_rule["feature_ids"],
                "tp": tp,
                "sl": sl,
                "capital_pct": cap,
            })

        rules.sort(key=lambda x: x["rule_id"])

        if len(rules) < self.min_rules:
            return {
                "direction": self.direction,
                "rules_set": []
            }

        return {
            "direction": self.direction,
            "rules_set": rules
        }

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.xl.copy(), self.xu.copy()

    def repair(self, chrom: np.ndarray) -> np.ndarray:
        return np.clip(chrom, self.xl, self.xu).astype(int)

    def is_valid(self, chrom: np.ndarray) -> bool:
        return np.all((chrom >= self.xl) & (chrom <= self.xu))

    def get_active_rule_ids(self, chrom: np.ndarray) -> List[int]:
        seen = set()
        for i in range(self.max_rules):
            base = i * self.genes_per_slot
            enabled = int(chrom[base])
            if enabled == 1:
                rid = int(chrom[base + 1])
                if 0 <= rid < self.n_rules:
                    seen.add(rid)
        return sorted(seen)
