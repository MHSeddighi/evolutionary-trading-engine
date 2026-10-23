"""Phase 2 fit package: persist/load the artifacts the live trader needs.

Two inputs, both produced upstream:

1. ruleset.json — the deployed strategy, written by you (or scripts/package_fit.py
   after an optimization run). Tolerant schema (all keys case-insensitive):

   {
     "direction": "long",                      // or "short"
     "rules": [
       {
         "rule_id": 12,                        // optional
         "name": "btc_pullback_breakout",      // optional human label
         "conditions": "[a] IS Weak Bullish AND [b] IS Very High",
         "tp": 1.5,                            // % take-profit
         "sl": 1.0,                            // % stop-loss
         "capital_pct": 5.0                    // % of free USDT per entry
       }
     ]
   }
   - "conditions" may also be a list of strings.
   - tp/sl/capital_pct default to the config defaults with a warning when absent.

2. Phase-1 fitted quantile bins: rules compare discretized values ("Weak Bullish"
   etc.), so live evaluation needs the EXACT same bin edges the miner/optimizer
   fitted on the research data. package_fit.py pickles them from a fresh run;
   load_discretizer() restores them here.
"""
import json
import logging
import pickle
from typing import Dict, List, Optional, Tuple

from src.core.discretizer import FeatureDiscretizer, MODE_CONFIG, parse_condition, normalize_direction

logger = logging.getLogger(__name__)

DEFAULT_TP = 1.0
DEFAULT_SL = 1.0
DEFAULT_CAP = 5.0

# Fit-package artifact version. v2 = adaptive per-feature bin counts + raw-scale
# quantile edges. v1 (fixed mode-based 2/3/5/10 classes, normalized-scale edges)
# is INCOMPATIBLE: load_discretizer refuses it instead of silently mismatching
# bin codes between research and live.
ARTIFACT_VERSION = 2


def parse_rule_conditions(spec) -> List[str]:
    """Accept 'a AND b', a list of condition strings, or mixed."""
    if isinstance(spec, str):
        return [c.strip() for c in spec.split(" AND ") if c.strip()]
    if isinstance(spec, (list, tuple)):
        conds = []
        for item in spec:
            conds.extend(parse_rule_conditions(item))
        return conds
    raise ValueError(f"Unsupported conditions spec: {spec!r}")


def load_ruleset(path: str) -> Dict:
    """Load and validate ruleset.json -> {"direction": str, "rules": [normalized dict]}."""
    with open(path, "r", encoding="utf8") as f:
        raw = json.load(f)
    # formats: {direction, rules|rules_set:[...]} or phase-1 fit
    # {direction, strategy:{rules_set:[...]}} (rules may also be strings)
    rules_in = raw.get("rules") or raw.get("rules_set") or \
        ((raw.get("strategy") or {}).get("rules_set")) \
        if isinstance(raw, dict) else raw
    if isinstance(raw, dict) and "direction" not in raw and "direction_match" not in raw:
        pass
    if not rules_in:
        raise ValueError("Ruleset file contains no rules ('rules' list expected)")
    direction = normalize_direction(raw.get("direction", "long"))
    rules = []
    for i, r in enumerate(rules_in):
        conds = parse_rule_conditions(
            r.get("conditions", r.get("conditions_list", r.get("rule"))))
        if not conds:
            logger.warning("Rule %d has no conditions — skipped", i)
            continue
        tp = r.get("tp", r.get("take_profit", DEFAULT_TP))
        sl = r.get("sl", r.get("stop_loss", DEFAULT_SL))
        cap = r.get("capital_pct", r.get("cap", DEFAULT_CAP))
        if tp <= 0 or sl <= 0 or cap <= 0:
            raise ValueError(f"Rule {i}: tp/sl/capital_pct must be > 0")
        rules.append({
            "rule_id": r.get("rule_id", i),
            "name": r.get("name", ""),
            "conditions": conds,
            "tp": float(tp), "sl": float(sl),
            "capital_pct": float(cap),
        })
    logger.info("Ruleset loaded: %d rules | direction=%s | %s",
                len(rules), direction, path)
    return {"direction": direction, "rules": rules}


def save_discretizer(disc: FeatureDiscretizer, path: str):
    """Pickle fitted bins/feature_info so live discretization matches research."""
    with open(path, "wb") as f:
        pickle.dump({"artifact_version": ARTIFACT_VERSION,
                     "feature_info": disc.feature_info,
                     "bins": disc.bins}, f)


def load_discretizer(path: str) -> FeatureDiscretizer:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    version = blob.get("artifact_version", 1)
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"fit_package.pkl at {path} is artifact_version={version} but this "
            f"build requires version={ARTIFACT_VERSION}. The feature-registry/"
            "binning representation changed — re-fit and re-package phase 1 "
            "before deploying; do not mix old rulesets with new bins.")
    disc = FeatureDiscretizer()
    disc.feature_info = blob["feature_info"]
    disc.bins = blob["bins"]
    disc.condition_lookup = disc.config.build_atomic_conditions() \
        if disc.feature_info else []
    return disc


class LiveRuleEvaluator:
    """Evaluates ruleset conditions against discretized live feature rows.

    Value-name -> code mapping comes from the persisted feature_info (the same
    MODE_CONFIG values the research pipeline used). Conditions for a feature
    missing in feature_info fail closed (warn once, rule skipped forever).
    """

    def __init__(self, disc: FeatureDiscretizer, ruleset: Dict):
        self.disc = disc
        self.ruleset = ruleset
        self.info_by_feature = {fi["col"]: fi for fi in disc.feature_info}
        self.active_rules: List[Dict] = []
        self.skipped_rules: List[Tuple[dict, str]] = []
        for rule in ruleset["rules"]:
            compiled = self._compile(rule)
            if compiled:
                self.active_rules.append(compiled)

    def _compile(self, rule: Dict) -> Optional[Dict]:
        """Attach per-condition (feature, required code) to the rule, or reject it."""
        ok = []
        for cond in rule["conditions"]:
            feat, value_name = parse_condition(cond)
            info = self.info_by_feature.get(feat)
            if info is None:
                self.skipped_rules.append((rule, f"unknown feature [{feat}]"))
                return None
            if feat == "symbol":
                # research categorical codes don't transfer to live symbol names —
                # such rules can only be symbol-specific in the research pool
                self.skipped_rules.append((rule, "symbol-conditional rule not portable"))
                return None
            values = info["values"]
            if value_name not in values:
                self.skipped_rules.append(
                    (rule, f"[{feat}] has no value '{value_name}'"))
                return None
            ok.append((feat, values.index(value_name)))
        return {**rule, "compiled": ok,
                "direction": self.ruleset["direction"]}

    def evaluate_row(self, codes: Dict[str, int], symbol_name: str) -> List[Dict]:
        """Returns triggered rules with per-condition code+value explanation."""
        triggered = []
        for rule in self.active_rules:
            details, passed = [], True
            for feat, code in rule["compiled"]:
                info = self.info_by_feature[feat]
                actual = codes.get(feat)
                m = (actual == code)
                details.append({
                    "feature": feat,
                    "required_code": code,
                    "required_value": info["values"][code],
                    "actual_code": actual,
                    "actual_value": (info["values"][actual]
                                     if actual is not None
                                     and 0 <= actual < len(info["values"])
                                     else None),
                    "match": m,
                })
                passed = passed and m
            if passed:
                triggered.append({**{k: rule[k] for k in
                                     ("rule_id", "name", "conditions", "tp", "sl",
                                      "capital_pct", "direction")},
                                  "condition_detail": details})
        return triggered
