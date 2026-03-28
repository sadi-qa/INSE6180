from __future__ import annotations
from collections import Counter
from typing import Dict, FrozenSet, List, Set, Tuple, Optional, Any

from src.eclat.eclat import build_vertical
from src.fpgrowth.fp_growth import mine_fp_growth
from src.eclat.eclat import eclat_counts

Itemset = FrozenSet[str]

class TriHybridResult:
    def __init__(self, frequent_itemsets: Dict[Itemset, float], stats: Dict[str, Any]):
        self.frequent_itemsets = frequent_itemsets
        self.stats = stats

def trihybrid_ppam(
    transactions_list: List[List[str]],
    transactions_set: List[Set[str]],
    minsup: float,
    density_threshold: float = 2.6
) -> TriHybridResult:
    """
    TriHybrid-PPAM (Pair-Pruned Adaptive Miner) - a hybrid algorithm combining ideas from:
      - Apriori: downward-closure (pruning logic)
      - Eclat: vertical tid-lists for fast pair counting
      - FP-growth: conditional mining without explicit candidate generation

    Steps:
      1) Count frequent 1-itemsets (support >= minsup).
      2) Build vertical tid-lists and compute frequent 2-itemsets (pairs) by intersection.
         This yields a "pair-pruning graph" that tells which items can co-occur frequently.
      3) For each item i (processed least frequent first), build a projected database of
         transactions containing i, keeping only items that:
           - are more frequent than i (FP-growth uniqueness)
           - form a frequent pair with i (pair pruning)
      4) Mine the projected database using FP-growth if it's dense, else Eclat.
         Then append i to every discovered pattern.

    Returns frequent itemsets with support fractions + useful stats.
    """
    n = len(transactions_list)
    minsup_count = max(1, int(minsup * n + 1e-9))

    # --- 1) frequent 1-itemsets ---
    item_counter = Counter()
    for t in transactions_list:
        item_counter.update(t)
    freq1_counts = {i: c for i, c in item_counter.items() if c >= minsup_count}
    if not freq1_counts:
        return TriHybridResult({}, {"minsup_count": minsup_count, "note": "No frequent items"})

    # global order by descending support (FP-growth ordering)
    order = [i for i, _ in sorted(freq1_counts.items(), key=lambda x: (-x[1], x[0]))]
    rank = {item: idx for idx, item in enumerate(order)}

    # initialize outputs with L1
    freq_out: Dict[Itemset, float] = {frozenset([i]): c/n for i, c in freq1_counts.items()}

    # --- 2) frequent pairs via vertical intersections (Eclat idea) ---
    vertical = build_vertical(transactions_set)  # item -> tidlist
    frequent_items = [i for i in order]  # already frequent
    # adjacency: for suffix item i, which prefix items (more frequent) can pair frequently with i
    adj: Dict[str, Set[str]] = {i: set() for i in frequent_items}
    pair_count = 0
    freq_pair_count = 0

    # compute pairs where prefix is more frequent (lower rank) than suffix
    for idx_i in range(len(frequent_items)):
        for idx_j in range(idx_i + 1, len(frequent_items)):
            a = frequent_items[idx_i]  # more frequent or lex earlier within tie
            b = frequent_items[idx_j]  # less frequent
            # pair support by intersection
            tids = vertical[a] & vertical[b]
            pair_count += 1
            if len(tids) >= minsup_count:
                freq_pair_count += 1
                # store in both directions; we'll query "more frequent items for suffix"
                adj[b].add(a)
                # also store the pair itself
                freq_out[frozenset([a, b])] = len(tids)/n

    # --- 3) mine projections for each suffix item (least frequent first) ---
    # process least frequent first: reverse of order list (since order is desc support)
    projections = 0
    mined_with_fp = 0
    mined_with_eclat = 0
    total_proj_tx = 0

    for suffix in reversed(order):
        allowed_prefix = adj.get(suffix, set())
        if not allowed_prefix:
            continue  # nothing can extend this suffix under pair pruning

        # build projected database: transactions containing suffix
        proj_list: List[List[str]] = []
        proj_set: List[Set[str]] = []

        for t_list, t_set in zip(transactions_list, transactions_set):
            if suffix not in t_set:
                continue
            filtered = [x for x in t_list if (x in allowed_prefix)]
            if not filtered:
                continue
            # FP-tree prefers ordering by global rank (more frequent first)
            filtered = sorted(set(filtered), key=lambda x: rank[x])
            proj_list.append(filtered)
            proj_set.append(set(filtered))

        if not proj_list:
            continue

        projections += 1
        total_proj_tx += len(proj_list)

        avg_len = sum(len(t) for t in proj_list) / len(proj_list)

        # --- 4) adaptive mining choice ---
        if avg_len >= density_threshold and len(proj_list) >= 5:
            mined_with_fp += 1
            # FP-growth returns support counts for patterns in proj DB (which corresponds to support of pattern+suffix)
            patterns_counts = mine_fp_growth(proj_list, minsup_count)
            for pat, sup_ct in patterns_counts.items():
                # pat is tuple of prefix items (more frequent than suffix)
                itemset = frozenset(pat + (suffix,))
                freq_out[itemset] = sup_ct / n
        else:
            mined_with_eclat += 1
            # Eclat expects set transactions; support fractions are relative to proj_db size,
            # but counts correspond to occurrences in proj_db = occurrences with suffix in original.
            # So convert back to global fraction by multiplying by |proj| / n.
            proj_n = len(proj_set)
            patterns_counts = eclat_counts(proj_set, minsup_count)
            for it, ct in patterns_counts.items():
                itemset = it | frozenset([suffix])
                freq_out[itemset] = ct / n

    stats = {
        "minsup": minsup,
        "minsup_count": minsup_count,
        "n_transactions": n,
        "n_freq_items": len(freq1_counts),
        "pairs_checked": pair_count,
        "freq_pairs_found": freq_pair_count,
        "projections_mined": projections,
        "mined_with_fp": mined_with_fp,
        "mined_with_eclat": mined_with_eclat,
        "avg_projected_transactions": (total_proj_tx / projections) if projections else 0.0,
        "density_threshold": density_threshold
    }

    return TriHybridResult(freq_out, stats)
