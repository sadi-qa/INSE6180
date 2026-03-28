from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set, Any

from src.apriori.apriori import apriori
from src.fpgrowth.fp_growth import mine_fp_growth
from src.eclat.eclat import eclat
from src.hybrid.trihybrid import trihybrid_ppam

Itemset = FrozenSet[str]

@dataclass
class A3MinerResult:
    frequent_itemsets: Dict[Itemset, float]
    chosen_engine: str
    stats: Dict[str, Any]

def a3mine(
    transactions_list: List[List[str]],
    transactions_set: List[Set[str]],
    minsup: float,
    strategy: str = "auto",
) -> A3MinerResult:
    """
    A3Miner (Adaptive 3-way Miner): an 'algorithm portfolio' for frequent itemset mining.

    It uses the three classic algorithms as engines (Apriori, FP-growth, Eclat) and can also
    consider the project-specific hybrid (TriHybrid-PPAM). The engine is selected by simple,
    explainable heuristics based on dataset shape and minsup.

    strategy:
      - "auto": choose engine based on heuristics
      - "apriori" / "fpgrowth" / "eclat" / "trihybrid": force engine
    """
    n = len(transactions_list)

    # basic shape stats
    unique_items = set()
    total_len = 0
    for t in transactions_list:
        tt = set(t)
        total_len += len(tt)
        unique_items.update(tt)
    m = len(unique_items)
    avg_len = total_len / n if n else 0.0
    density = (avg_len / m) if m else 0.0

    engine = strategy
    reason = ""

    if strategy == "auto":
        # Practical heuristics (easy to justify in a report):
        #  - When the item universe is small-to-moderate AND baskets are short, vertical tid-lists are cheap -> Eclat.
        #  - When baskets are very long or the item universe is large, FP-tree compression helps -> FP-growth.
        #  - When minsup is very high, Apriori's candidate frontier stays small -> Apriori.
        if minsup >= 0.45:
            engine = "apriori"
            reason = "High minsup: few frequent itemsets; Apriori's candidate frontier stays small."
        elif m <= 500 and avg_len <= 10:
            engine = "eclat"
            reason = "Small/medium item universe with short baskets: tid-list intersections are efficient."
        elif avg_len >= 12 or m >= 1000:
            engine = "fpgrowth"
            reason = "Long baskets or large item universe: FP-tree compression and conditional mining reduce overhead."
        else:
            engine = "trihybrid"
            reason = "Mixed profile: pair-pruned projections (TriHybrid-PPAM) balance pruning and mining."
    else:
        reason = "Manual selection"

    # run selected engine
    if engine == "apriori":
        freq = apriori(transactions_set, minsup)
    elif engine == "fpgrowth":
        minsup_count = max(1, int(minsup * n + 1e-9))
        counts = mine_fp_growth(transactions_list, minsup_count)
        freq = {frozenset(k): v / n for k, v in counts.items()}
    elif engine == "eclat":
        freq = eclat(transactions_set, minsup)
    elif engine == "trihybrid":
        tri = trihybrid_ppam(transactions_list, transactions_set, minsup)
        freq = tri.frequent_itemsets
    else:
        raise ValueError(f"Unknown engine: {engine}")

    stats = {
        "n": n,
        "m": m,
        "avg_len": avg_len,
        "density": density,
        "minsup": minsup,
        "engine": engine,
        "reason": reason
    }
    return A3MinerResult(freq, engine, stats)
