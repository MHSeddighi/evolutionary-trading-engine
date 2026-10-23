'''Rule pool (mined-rule CSV loader with KMeans clustering) and pool-based
pymoo sampling. Merged from legacy plugins; logic unchanged.'''
import csv
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pymoo.core.sampling import Sampling
from pymoo.util import default_random_state

_GLOBAL_RULES: List[Dict] = []
_GLOBAL_CLUSTER_LABELS: Optional[np.ndarray] = None
_GLOBAL_CLUSTERS: Dict[int, List[int]] = {}
_GLOBAL_FEATURES: Optional[np.ndarray] = None
_GLOBAL_SCALER = None



class RulePool:

    def __init__(self, csv_path: str, n_clusters: int = 30, random_state: int = 42):
        self.csv_path = csv_path
        self.n_clusters = n_clusters
        self.random_state = random_state

        if len(_GLOBAL_RULES) == 0:
            self._load_csv(csv_path)
            self._build_features()
            self._build_clusters(n_clusters, random_state)

    @property
    def rules(self) -> List[Dict]:
        return _GLOBAL_RULES

    @property
    def cluster_labels(self) -> Optional[np.ndarray]:
        return _GLOBAL_CLUSTER_LABELS

    @property
    def clusters(self) -> Dict[int, List[int]]:
        return _GLOBAL_CLUSTERS

    @property
    def features(self) -> Optional[np.ndarray]:
        return _GLOBAL_FEATURES

    @property
    def scaler(self):
        return _GLOBAL_SCALER

    def _parse_conditions(self, row: Dict) -> List[str]:
        """Parse conditions from any column name variation."""
        for key in ['conditions', 'condition', 'conds', 'rule_conditions']:
            val = row.get(key, '')
            if val:
                break

        if not val:
            return []

        # Try different separators
        for sep in [',', ' & ', ';', ' AND ']:
            if sep in val:
                conds = [c.strip() for c in val.split(sep) if c.strip()]
                break
        else:
            conds = [val.strip()] if val.strip() else []

        # Remove symbol conditions and clean
        conds = [c for c in conds if not c.lower().startswith('[symbol]')]
        return conds

    def _safe_float(self, row: Dict, *keys, default=0.0) -> float:
        """Safely extract float from row using multiple possible column names."""
        for key in keys:
            try:
                val = row.get(key)
                if val is not None and val != '':
                    return float(val)
            except (ValueError, TypeError):
                continue
        return default

    def _safe_int(self, row: Dict, *keys, default=0) -> int:
        """Safely extract int from row using multiple possible column names."""
        for key in keys:
            try:
                val = row.get(key)
                if val is not None and val != '':
                    return int(float(val))
            except (ValueError, TypeError):
                continue
        return default

    def _load_csv(self, path: str, allow_missing=True):
        global _GLOBAL_RULES

        if allow_missing and not os.path.exists(path):
            print(f"⚠️  Rule pool CSV not found (cold start): {path} — pool starts empty")
            return

        with open(path, "r", encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                print(f"⚠️  Empty or invalid CSV: {path}")
                return

            # Normalize fieldnames to lowercase
            fieldnames = [f.strip().lower().replace(' ', '_') for f in reader.fieldnames]

            for row in reader:
                # Normalize row keys
                normalized_row = {}
                for i, (key, value) in enumerate(row.items()):
                    if i < len(fieldnames):
                        normalized_row[fieldnames[i]] = value
                    else:
                        normalized_row[key.strip().lower().replace(' ', '_')] = value

                conds = self._parse_conditions(normalized_row)
                if not conds:
                    continue

                pnl = self._safe_float(normalized_row, 'pnl', 'train_pnl', 'test_pnl', 'full_pnl', 'total_return_pct',
                                       'return')
                dd = self._safe_float(normalized_row, 'dd', 'train_dd', 'test_dd', 'full_dd', 'max_drawdown_pct',
                                      'drawdown', default=100.0)
                pf = self._safe_float(normalized_row, 'pf', 'train_pf', 'test_pf', 'full_pf', 'profit_factor')
                wr = self._safe_float(normalized_row, 'wr', 'train_wr', 'test_wr', 'win_rate', 'winrate')
                trades = self._safe_int(normalized_row, 'trades', 'train_trades', 'test_trades', 'full_trades',
                                        'executed_trades', 'n_trades')
                tp = self._safe_float(normalized_row, 'tp', 'take_profit', default=3.0)
                sl = self._safe_float(normalized_row, 'sl', 'stop_loss', default=1.5)
                capital_pct = self._safe_float(normalized_row, 'capital_pct', 'cap', default=1.0)
                gini = self._safe_float(normalized_row, 'gini', 'train_gini', 'full_gini', default=0.5)
                score = self._safe_float(normalized_row, 'score')
                n_conds = self._safe_int(normalized_row, 'n_conds', default=len(conds))

                _GLOBAL_RULES.append({
                    "conditions": conds,
                    "tp": tp,
                    "sl": sl,
                    "capital_pct": capital_pct,
                    "metrics": {
                        "return": pnl,
                        "drawdown": dd,
                        "pf": pf,
                        "winrate": wr,
                        "trades": trades,
                        "gini": gini,
                        "score": score,
                        "n_conds": n_conds,
                    },
                })

        print(f"📚 Loaded {len(_GLOBAL_RULES)} rules from {path}")

    def _build_features(self):
        global _GLOBAL_FEATURES, _GLOBAL_SCALER

        if len(_GLOBAL_RULES) == 0:
            _GLOBAL_FEATURES = np.array([])
            return

        raw = []
        for r in _GLOBAL_RULES:
            m = r["metrics"]
            n_conds = len(r["conditions"])
            raw.append([
                m["return"],
                m["drawdown"],
                m["pf"],
                m["winrate"],
                np.log1p(max(1, m["trades"])),
                m["return"] / (m["drawdown"] + 1e-8),
                m["pf"] * m["winrate"] / 100,
                np.log1p(max(1, m["trades"])) * m["winrate"] / 100,
                r["tp"],
                r["sl"],
                r["tp"] / (r["sl"] + 1e-8),
                n_conds,
                m["return"] / (np.log1p(max(1, m["trades"])) + 1e-8),
                m.get("gini", 0.5),
                m.get("score", 0),
            ])

        raw = np.asarray(raw, dtype=np.float32)

        # Handle NaN/Inf
        raw = np.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)

        _GLOBAL_SCALER = StandardScaler()
        _GLOBAL_FEATURES = _GLOBAL_SCALER.fit_transform(raw)

    def _build_clusters(self, n_clusters: int, random_state: int):
        global _GLOBAL_CLUSTER_LABELS, _GLOBAL_CLUSTERS

        if len(_GLOBAL_RULES) < 2 or _GLOBAL_FEATURES is None or len(_GLOBAL_FEATURES) == 0:
            return

        n_clusters = min(n_clusters, len(_GLOBAL_RULES))
        n_clusters = max(2, min(3, n_clusters))

        _GLOBAL_CLUSTER_LABELS = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            init="k-means++",
            max_iter=500,
        ).fit_predict(_GLOBAL_FEATURES)

        _GLOBAL_CLUSTERS = {}
        for idx, label in enumerate(_GLOBAL_CLUSTER_LABELS):
            _GLOBAL_CLUSTERS.setdefault(int(label), []).append(idx)

        sizes = [len(v) for v in _GLOBAL_CLUSTERS.values()]
        print(f"🔬 Clusters: {n_clusters} | Min: {min(sizes)}, Max: {max(sizes)}, Std: {np.std(sizes):.0f}")

    def get_rule(self, idx: int) -> Dict:
        return _GLOBAL_RULES[idx]

    def get_rule_features(self, idx: int) -> np.ndarray:
        return _GLOBAL_FEATURES[idx]

    def get_cluster(self, cluster_id: int) -> List[int]:
        return _GLOBAL_CLUSTERS.get(cluster_id, [])

    def get_rule_cluster(self, rule_idx: int) -> int:
        return int(_GLOBAL_CLUSTER_LABELS[rule_idx])

    def get_cluster_ids(self) -> List[int]:
        return list(_GLOBAL_CLUSTERS.keys())

    def get_cluster_center(self, cluster_id: int) -> np.ndarray:
        indices = _GLOBAL_CLUSTERS.get(cluster_id, [])
        if not indices:
            return np.array([])
        return np.mean(_GLOBAL_FEATURES[indices], axis=0)

    def sample_cluster_rule(self, cluster_id: int, rng: np.random.Generator) -> int:
        cluster_rules = _GLOBAL_CLUSTERS.get(cluster_id, [])
        if not cluster_rules:
            return rng.integers(0, len(_GLOBAL_RULES))
        return rng.choice(cluster_rules)

    def get_best_rule_from_cluster(self, cluster_id: int, metric: str = "return") -> int:
        indices = _GLOBAL_CLUSTERS.get(cluster_id, [])
        if not indices:
            return -1
        return max(indices, key=lambda i: _GLOBAL_RULES[i]["metrics"].get(metric, -1e18))

    def get_rules_by_metric(self, metric: str = "return", top_k: int = 100) -> List[Dict]:
        return sorted(_GLOBAL_RULES, key=lambda x: x["metrics"].get(metric, -1e18), reverse=True)[:top_k]

    def __len__(self) -> int:
        return len(_GLOBAL_RULES)

    def __getitem__(self, idx: int) -> Dict:
        return _GLOBAL_RULES[idx]


class PoolBasedSampling(Sampling):

    def __init__(self, decoder=None, rule_pool=None, condition_lookup=None,
                 pool_ratio=0.5, seed_ratio=0.0, seed_dir=None):
        super().__init__()
        self.decoder = decoder
        self.rule_pool = rule_pool
        self.condition_lookup = condition_lookup
        self.pool_ratio = pool_ratio
        self.seed_ratio = seed_ratio
        self.random_ratio = 1.0 - pool_ratio - seed_ratio

        self.seeds = []
        if seed_dir and os.path.exists(seed_dir) and seed_ratio > 0:
            self._load_seeds(seed_dir)

        self.cond_string_to_id = {c["condition_string"]: c["id"] for c in condition_lookup} if condition_lookup else {}
        self.used_pool_ids = set()
        self.pool_id_usage = {}
        self._build_scores()

    def _load_seeds(self, seed_dir):
        seed_path = Path(seed_dir)
        json_files = list(seed_path.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                rules_set = data.get("strategy", {}).get("rules_set", [])
                if rules_set:
                    self.seeds.append({
                        "rules_set": rules_set,
                        "direction": data.get("strategy", {}).get("direction", "long"),
                        "source": json_file.name,
                        "rank": data.get("rank", 999),
                        "train": data.get("train", {}),
                        "test": data.get("test", {}),
                    })
            except Exception:
                pass
        self.seeds.sort(key=lambda x: x.get("rank", 999))
        print(f"   Total seed strategies loaded: {len(self.seeds)}")

    def _build_scores(self):
        self.rule_scores = np.zeros(len(self.rule_pool.rules))
        for i, rule in enumerate(self.rule_pool.rules):
            m = rule["metrics"]
            self.rule_scores[i] = (
                                          max(m.get("pf", 0.01), 0.01) *
                                          (m.get("return", 0) + 1) *
                                          np.log1p(m.get("trades", 1))
                                  ) / (m.get("drawdown", 1) + 1)

    def _find_rule_id_by_conditions(self, conditions):
        for rule_id, pool_rule in enumerate(self.rule_pool.rules):
            if set(pool_rule.get("conditions", [])) == set(conditions):
                return rule_id
        return -1

    def _sample_single_rule(self, rng, elite_bias=True):
        pool_size = len(self.rule_pool.rules)
        if pool_size == 0:
            return -1
        if elite_bias and rng.random() < 0.3:
            elite_idx = rng.choice(np.argsort(self.rule_scores)[-max(50, pool_size // 10):])
            return int(elite_idx)
        return rng.integers(0, pool_size)

    def _build_chromosome(self, problem, rule_ids, rng, n_target=None):
        max_rules = self.decoder.max_rules
        genes_per_slot = self.decoder.genes_per_slot
        # honor the requested ruleset size (min..max) instead of always
        # padding to max_rules, so 3..6-rule strategies are all explored
        if n_target is not None:
            n_target = max(int(n_target), self.decoder.min_rules)
            max_rules = min(max_rules, n_target)

        cap_steps = self.decoder.cap_steps
        tp_steps = self.decoder.tp_offset_steps
        sl_steps = self.decoder.sl_offset_steps

        chrom = np.zeros(problem.n_var, dtype=np.int32)

        seen = set()
        slot = 0

        for rid in rule_ids:
            if slot >= max_rules:
                break
            if 0 <= rid < len(self.rule_pool.rules) and rid not in seen:
                base = slot * genes_per_slot
                chrom[base] = 1  # Enabled — every sampled rule starts active so the
                #              # search explores the FULL 3..6-rule space, not just 2
                chrom[base + 1] = rid
                chrom[base + 2] = rng.integers(0, tp_steps)
                chrom[base + 3] = rng.integers(0, sl_steps)
                chrom[base + 4] = rng.integers(0, cap_steps)

                self.used_pool_ids.add(rid)
                self.pool_id_usage[rid] = self.pool_id_usage.get(rid, 0) + 1
                seen.add(rid)
                slot += 1

        tries, max_tries = 0, max_rules * 20
        while slot < max_rules and tries < max_tries:
            rid = self._sample_single_rule(rng, elite_bias=False)
            tries += 1
            if rid not in seen and rid >= 0:
                base = slot * genes_per_slot
                chrom[base] = 1
                chrom[base + 1] = rid
                chrom[base + 2] = rng.integers(0, tp_steps)
                chrom[base + 3] = rng.integers(0, sl_steps)
                chrom[base + 4] = rng.integers(0, cap_steps)

                self.used_pool_ids.add(rid)
                self.pool_id_usage[rid] = self.pool_id_usage.get(rid, 0) + 1
                seen.add(rid)
                slot += 1

        return chrom

    def _sample_from_pool(self, n, rng, problem):
        chroms = []
        for _ in range(n):
            # START at the maximum ruleset size; the annealed rule_cap
            # constraint + mutation trim it down over the generations
            n_rules = self.decoder.max_rules
            rule_ids = [self._sample_single_rule(rng) for _ in range(n_rules)]
            chroms.append(self._build_chromosome(problem, rule_ids, rng, n_target=n_rules))
        return chroms

    def _sample_from_seeds(self, n, rng, problem):
        chroms = []
        if not self.seeds:
            return chroms
        for _ in range(n):
            seed = rng.choice(self.seeds)
            rules_set = seed.get("rules_set", [])
            rule_ids = []

            for rule in rules_set:
                conditions = rule.get("conditions", [])
                rid = self._find_rule_id_by_conditions(conditions)
                if rid >= 0:
                    rule_ids.append(rid)

            if len(rule_ids) >= self.decoder.min_rules:
                chroms.append(self._build_chromosome(problem, rule_ids[:self.decoder.max_rules], rng))
            else:
                while len(rule_ids) < self.decoder.min_rules:
                    rid = self._sample_single_rule(rng)
                    if rid not in rule_ids:
                        rule_ids.append(rid)
                chroms.append(self._build_chromosome(problem, rule_ids, rng))

        return chroms

    def _sample_random(self, n, rng, problem):
        chroms = []
        pool_size = len(self.rule_pool.rules)
        if pool_size == 0:
            return chroms
        for _ in range(n):
            n_rules = rng.integers(self.decoder.min_rules, self.decoder.max_rules + 1)
            rule_ids = [int(rng.integers(0, pool_size)) for _ in range(n_rules)]
            chroms.append(self._build_chromosome(problem, rule_ids, rng))
        return chroms

    def get_stats(self):
        if not self.pool_id_usage:
            return
        total = sum(self.pool_id_usage.values())
        unique = len(self.used_pool_ids)
        print(f"\n📊 Initialization Stats:")
        print(f"   Pool rules used: {unique} unique ({total} total)")
        if self.seeds:
            print(f"   Seed strategies: {len(self.seeds)}")

    @default_random_state
    def _do(self, problem, n_samples, random_state=None, **kwargs):
        rng = random_state

        n_pool = int(n_samples * self.pool_ratio)
        n_seed = int(n_samples * self.seed_ratio)
        n_random = n_samples - n_pool - n_seed

        chromosomes = []

        if n_pool > 0:
            chromosomes.extend(self._sample_from_pool(n_pool, rng, problem))
        if n_seed > 0:
            chromosomes.extend(self._sample_from_seeds(n_seed, rng, problem))
        if n_random > 0:
            chromosomes.extend(self._sample_random(n_random, rng, problem))

        # fallback for cold start (empty pool generates no chromosomes)
        while len(chromosomes) < n_samples:
            chromosomes.extend(self._sample_random(n_samples - len(chromosomes), rng, problem))
            break

        self.get_stats()
        return np.asarray(chromosomes[:n_samples], dtype=np.int32)


def make_synthetic_pool(condition_lookup, features, max_single=96,
                        cov_min=0.05, cov_max=0.5, tp=1.0, sl=0.5, cap=1.0):
    """Cold-startRulePool: bootstrap from single conditions with usable coverage.

    Fills the module-level pool so PoolBasedSampling/Decoder work before any
    miner iteration has produced real rules.
    """
    from src.core.backtest import apply_dynamic_rule
    global _GLOBAL_RULES
    if len(_GLOBAL_RULES) >= max_single:
        return
    rng = np.random.default_rng(42)
    # shuffle conditions to diversify categories
    cond_list = list(condition_lookup)
    rng.shuffle(cond_list)
    n = len(next(iter(features.values())))
    kept = 0
    for c in cond_list:
        if kept >= max_single:
            break
        try:
            cov = float(apply_dynamic_rule(features, c["condition_string"]).mean())
        except ValueError:
            continue
        if cov_min <= cov <= cov_max:
            _GLOBAL_RULES.append({
                "conditions": [c["condition_string"]],
                "tp": tp, "sl": sl, "capital_pct": cap,
                "metrics": {"return": 0.0, "drawdown": 1.0, "pf": 1.0,
                            "winrate": 50.0, "trades": cov * n, "gini": 0.5, "score": 0.0},
            })
            kept += 1
    print(f"🧬 Synthetic cold-start pool: {len(_GLOBAL_RULES)} single-rule entries")


# monkey-patch RulePool with a loader-free constructor helper
def build_pool(conditions_lookup=None, features=None, csv_path=None, n_clusters=30,
               min_needed=3):
    global _GLOBAL_RULES
    RulePool(csv_path or "__none__", n_clusters=n_clusters)   # loads real rules if present
    # drop rules referencing features pruned from the current dataset
    if conditions_lookup is not None and _GLOBAL_RULES:
        import re
        def _feat_of(cond: str):
            m = re.match(r"\s*\[([^\]]+)\]", cond)
            return m.group(1) if m else None
        allowed = {(_feat_of(c) or "") for r in _GLOBAL_RULES for c in r["conditions"]}
        valid = {cmap["feature_name"] for cmap in conditions_lookup}
        before = len(_GLOBAL_RULES)
        _GLOBAL_RULES = [r for r in _GLOBAL_RULES
                         if all((_feat_of(c) in valid) for c in r["conditions"])]
        after = len(_GLOBAL_RULES)
        if after < before:
            print(f"🧹 Dropped {before - after} pool rules using pruned features "
                  f"({after} kept)")
    if len(_GLOBAL_RULES) < min_needed and conditions_lookup is not None and features is not None:
        make_synthetic_pool(conditions_lookup, features)      # top-up cold start
    np.random.shuffle(_GLOBAL_RULES)
    class _View:
        rules = _GLOBAL_RULES

        def __len__(self):
            return len(_GLOBAL_RULES)
    return _View()
