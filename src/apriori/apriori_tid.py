from __future__ import annotations
from collections import defaultdict
from typing import Dict, FrozenSet, List, Set, Tuple

Itemset = FrozenSet[str]
TidEntry = Tuple[int, Set[Itemset]]  # (tid, set of (k-1)-candidates present in transaction)

def find_frequent_1_itemsets(transactions: List[Set[str]], minsup: float) -> Dict[Itemset, float]:
    counts = defaultdict(int)
    for t in transactions:
        for item in t:
            counts[item] += 1
    n = len(transactions)
    return {frozenset([i]): ct/n for i, ct in counts.items() if (ct/n) >= minsup}

def apriori_gen(L_prev: Dict[Itemset, float], k: int):
    L_list = sorted([tuple(sorted(s)) for s in L_prev.keys()])
    candidates = set()
    for i in range(len(L_list)):
        for j in range(i+1, len(L_list)):
            a, b = L_list[i], L_list[j]
            if a[:k-2] == b[:k-2]:
                candidates.add(frozenset(a) | frozenset(b))
    return candidates

def prune_candidates(candidates, L_prev):
    out = set()
    for c in candidates:
        ok = True
        for item in c:
            if (c - frozenset([item])) not in L_prev:
                ok = False
                break
        if ok:
            out.add(c)
    return out

def build_E1(transactions: List[Set[str]]) -> List[TidEntry]:
    E1 = []
    for tid, t in enumerate(transactions):
        E1.append((tid, {frozenset([i]) for i in t}))
    return E1

def apriori_tid(transactions: List[Set[str]], minsup: float) -> Dict[Itemset, float]:
    """
    AprioriTid (Paper 1, Section 2.2):
      - scan database only in pass 1
      - later passes use per-transaction encoding of candidate sets
    Returns dict itemset -> support fraction.
    """
    n = len(transactions)
    minsup_count = max(1, int(minsup * n + 1e-9))

    all_freq: Dict[Itemset, float] = {}

    L_prev = find_frequent_1_itemsets(transactions, minsup)
    all_freq.update(L_prev)

    E_prev = build_E1(transactions)
    k = 2

    while L_prev:
        Ck = apriori_gen(L_prev, k)
        Ck = prune_candidates(Ck, L_prev)
        if not Ck:
            break

        counts = defaultdict(int)
        E_k: List[TidEntry] = []

        for tid, present_prev in E_prev:
            present_k = set()
            for c in Ck:
                # c exists if all (k-1)-subsets existed last pass
                ok = True
                for item in c:
                    if (c - frozenset([item])) not in present_prev:
                        ok = False
                        break
                if ok:
                    present_k.add(c)
                    counts[c] += 1
            if present_k:
                E_k.append((tid, present_k))

        Lk = {c: ct/n for c, ct in counts.items() if ct >= minsup_count}
        if not Lk:
            break
        all_freq.update(Lk)

        E_prev = E_k
        L_prev = Lk
        k += 1

    return all_freq
