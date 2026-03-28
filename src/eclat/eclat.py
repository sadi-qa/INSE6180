from __future__ import annotations
from typing import Dict, FrozenSet, List, Set, Tuple

Itemset = FrozenSet[str]
TidList = FrozenSet[int]

def build_vertical(transactions: List[Set[str]]) -> Dict[str, TidList]:
    vertical: Dict[str, Set[int]] = {}
    for tid, t in enumerate(transactions):
        for item in t:
            vertical.setdefault(item, set()).add(tid)
    return {i: frozenset(tids) for i, tids in vertical.items()}

def eclat_counts(transactions: List[Set[str]], minsup_count: int) -> Dict[Itemset, int]:
    """
    Eclat frequent itemset mining in vertical format (Paper 3).
    Returns dict itemset -> support COUNT (not fraction).
    """
    vertical = build_vertical(transactions)
    items = [(item, tids) for item, tids in vertical.items() if len(tids) >= minsup_count]
    items.sort(key=lambda x: (len(x[1]), x[0]))

    freq: Dict[Itemset, int] = {frozenset([i]): len(tids) for i, tids in items}

    def dfs(prefix: Tuple[str, ...], prefix_tids: TidList, suffix: List[Tuple[str, TidList]]):
        for idx in range(len(suffix)):
            item, tids = suffix[idx]
            new_items = prefix + (item,)
            new_tids = prefix_tids & tids
            if len(new_tids) >= minsup_count:
                freq[frozenset(new_items)] = len(new_tids)
                new_suffix: List[Tuple[str, TidList]] = []
                for j in range(idx+1, len(suffix)):
                    item2, tids2 = suffix[j]
                    inter = new_tids & tids2
                    if len(inter) >= minsup_count:
                        new_suffix.append((item2, inter))
                dfs(new_items, new_tids, new_suffix)

    for i, (item, tids) in enumerate(items):
        dfs((item,), tids, items[i+1:])

    return freq

def eclat(transactions: List[Set[str]], minsup: float) -> Dict[Itemset, float]:
    """
    Convenience wrapper returning support fractions.
    """
    n = len(transactions)
    minsup_count = max(1, int(minsup * n + 1e-9))
    counts = eclat_counts(transactions, minsup_count)
    return {k: v/n for k, v in counts.items()}
